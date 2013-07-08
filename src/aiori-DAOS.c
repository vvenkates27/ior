/* -*- mode: c; c-basic-offset: 8; indent-tabs-mode: nil; -*-
 * vim:expandtab:shiftwidth=8:tabstop=8:
 */
/******************************************************************************\
*                                                                              *
*        Copyright (c) 2013, Intel Corporation.                                *
*      See the file COPYRIGHT for a complete copyright notice and license.     *
*                                                                              *
********************************************************************************
*
* Implement of abstract I/O interface for DAOS.
*
\******************************************************************************/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdint.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <daos_api.h>

#include "ior.h"
#include "aiori.h"
#include "iordef.h"

/**************************** P R O T O T Y P E S *****************************/

static void *DAOS_Create(char *, IOR_param_t *);
static void *DAOS_Open(char *, IOR_param_t *);
static IOR_offset_t DAOS_Xfer(int, void *, IOR_size_t *,
                              IOR_offset_t, IOR_param_t *);
static void DAOS_Close(void *, IOR_param_t *);
static void DAOS_Delete(char *, IOR_param_t *);
static void DAOS_SetVersion(IOR_param_t *);
static void DAOS_Fsync(void *, IOR_param_t *);
static IOR_offset_t DAOS_GetFileSize(IOR_param_t *, MPI_Comm, char *);

/************************** D E C L A R A T I O N S ***************************/

ior_aiori_t daos_aiori = {
        "DAOS",
        DAOS_Create,
        DAOS_Open,
        DAOS_Xfer,
        DAOS_Close,
        DAOS_Delete,
        DAOS_SetVersion,
        DAOS_Fsync,
        DAOS_GetFileSize
};

struct fileDescriptor {
        daos_handle_t container;
        daos_handle_t object;
        daos_epoch_t  epoch;
};

static daos_handle_t      eventQueue;
static struct daos_event *events;
static struct daos_iod   *iods;
static struct daos_mmd   *mmds;
static int                nEvents;
static unsigned int      *targets;
static int                nTargets;
static int                initialized;

/***************************** F U N C T I O N S ******************************/

#define DCHECK(rc, format, ...)                                         \
do {                                                                    \
        int _rc = (rc);                                                 \
                                                                        \
        if (_rc < 0) {                                                  \
                fprintf(stdout, "ior ERROR (%s:%d): %d: %s: "           \
                        format"\n", __FILE__, __LINE__, rank,           \
                        strerror(-_rc), ##__VA_ARGS__);                 \
                fflush(stdout);                                         \
                MPI_Abort(MPI_COMM_WORLD, -1);                          \
        }                                                               \
} while (0);

static void SysInfoInit(void)
{
        daos_handle_t        sysContainer;
        struct daos_location loc;
        struct daos_loc_key *lks;
        unsigned int         lkn;
        int                  i;
        int                  rc;

        if (rank == 0) {
                rc = daos_sys_open(getenv("DAOS_POSIX"), &sysContainer,
                                          NULL /* synchronous */);
                DCHECK(rc, "Failed to open system container");

                loc.lc_cage = DAOS_LOC_UNKNOWN;
                rc = daos_sys_query(sysContainer, &loc, 0 /* whole tree */,
                                    &lkn, NULL /* size query */,
                                    NULL /* synchronous */);
                DCHECK(rc, "Failed to get size of location keys");

                lks = malloc((sizeof *lks) * lkn);
                if (lks == NULL)
                        ERR("Failed to allocate location keys");

                rc = daos_sys_query(sysContainer, &loc, 0 /* whole tree */,
                                    &lkn, lks, NULL /* synchronous */);
                DCHECK(rc, "Failed to get location keys");

                nTargets = 0;
                for (i = 0; i < lkn; i++)
                        if (lks[i].lk_type == DAOS_LOC_TYP_TARGET)
                                nTargets++;
        }

        MPI_Bcast(&nTargets, 1, MPI_INT, 0, MPI_COMM_WORLD);

        targets = malloc((sizeof *targets) * nTargets);
        if (targets == NULL)
                ERR("Failed to allocate target array");

        if (rank == 0) {
                unsigned int   *t;

                t = targets;
                for (i = 0; i < lkn; i++)
                        if (lks[i].lk_type == DAOS_LOC_TYP_TARGET)
                                *t++ = lks[i].lk_id;

                free(lks);

                rc = daos_sys_close(sysContainer, NULL /* synchronous */);
                DCHECK(rc, "Failed to get location keys");
        }

        MPI_Bcast(targets, nTargets, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}

static SysInfoFini(void)
{
        free(targets);
}

static void Init(void)
{
        int rc;

        rc = daos_posix_init();
        DCHECK(rc, "Failed to initialize daos-posix");

	rc = daos_eq_create(&eventQueue);
        DCHECK(rc, "Failed to create event queue");

        SysInfoInit();
}

static void Fini(void)
{
        int rc;

        SysInfoFini();

	rc = daos_eq_destroy(eventQueue);
        DCHECK(rc, "Failed to destroy event queue");

        daos_posix_finalize();
}

static void ShardAdd(daos_handle_t container, daos_epoch_t *epoch,
                     IOR_param_t *param)
{
        epoch->seq++;

        if (rank == 0) {
                int i;
                int rc;

                rc = daos_shard_add(container, *epoch, nTargets, targets,
                                    NULL /* ? */, NULL /* synchronous */);
                DCHECK(rc, "Failed to create shards");

                rc = daos_epoch_commit(container, *epoch, 1 /* sync */, NULL,
                                       NULL /* synchronous */);
                DCHECK(rc, "Failed to commit shard creation");
        }
}

static void ContainerOpen(char *testFileName, IOR_param_t *param,
                          daos_handle_t *container, daos_epoch_t *epoch)
{
        unsigned char *buffer;
        unsigned int   size;
        int            rc;

        if (rank == 0) {
                struct daos_epoch_info info;
                unsigned int           dMode;

                if (param->open == WRITE)
                        dMode = DAOS_COO_RW | DAOS_COO_CREATE;
                else
                        dMode = DAOS_COO_RO;

                rc = daos_container_open(testFileName, dMode, param->numTasks,
                                         NULL /* ignore status */, container,
                                         NULL /* synchronous */);
                DCHECK(rc, "Failed to open container %s", testFileName);

                rc = daos_epoch_query(*container, &info,
                                      NULL /* synchronous */);
                DCHECK(rc, "Failed to get epoch info from %s", testFileName);

                *epoch = info.epi_hce;
        }

        if (param->open == WRITE)
                ShardAdd(*container, epoch, param);

        if (param->numTasks == 1)
                return;

        if (rank == 0) {
                rc = daos_local2global(*container, NULL, &size);
                DCHECK(rc, "Failed to get global container handle size");
        }

        MPI_CHECK(MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, param->testComm),
                  "Failed to broadcast size");

        buffer = malloc(size + sizeof epoch->seq);
        if (buffer == NULL)
                ERR("Failed to allocate message buffer");

        if (rank == 0) {
                rc = daos_local2global(*container, buffer, &size);
                DCHECK(rc, "Failed to get global container handle");

                memmove(buffer + size, &epoch->seq, sizeof epoch->seq);
        }

        MPI_CHECK(MPI_Bcast(buffer, size + sizeof epoch->seq, MPI_BYTE, 0,
                            param->testComm), "Failed to broadcast message");

        if (rank != 0) {
                rc = daos_global2local(buffer, size, container,
                                       NULL /* sychronous */);
                DCHECK(rc, "Failed to get local container handle");

                memmove(&epoch->seq, buffer + size, sizeof epoch->seq);
        }
}

static void ContainerClose(daos_handle_t container, daos_epoch_t *epoch,
                           IOR_param_t *param)
{
        int rc;

        rc = daos_container_close(container, NULL /* synchronous */);
        DCHECK(rc, "Failed to close container");
}

static void ObjectOpen(daos_handle_t container, daos_handle_t *object,
                       daos_epoch_t *epoch, IOR_param_t *param)
{
        daos_obj_id_t oid;
        unsigned int  mode;
        int           rc;

        oid.o_id_hi = rank % nTargets;
        oid.o_id_lo = rank / nTargets;

        mode = DAOS_OBJ_IO_SEQ;
        if (param->open == WRITE)
                mode |= DAOS_OBJ_RW | DAOS_OBJ_EXCL;
        else
                mode |= DAOS_OBJ_RO;

        if (param->verbose > VERBOSE_2)
                printf("process %d opening object <%llu, %llu> with mode %x\n",
                       rank, oid.o_id_hi, oid.o_id_lo, mode);

        rc = daos_object_open(container, oid, mode, object,
                              NULL /* synchronous */);
        DCHECK(rc, "Failed to open object");
}

static void ObjectClose(daos_handle_t object)
{
        int rc;

        rc = daos_object_close(object, NULL /* synchronous */);
        DCHECK(rc, "Failed to close object");
}

static AIOInit(IOR_param_t *param)
{
        nEvents = param->aiosPerTransfer;

        events = malloc((sizeof *events) * nEvents);
        if (events == NULL)
                ERR("Failed to allocate event array");

        iods = malloc(((sizeof *iods) + (sizeof iods->iod_frag[0])) * nEvents);
        if (iods == NULL)
                ERR("Failed to allocate iod array");

        mmds = malloc(((sizeof *mmds) + (sizeof mmds->mmd_frag[0])) * nEvents);
        if (mmds == NULL)
                ERR("Failed to allocate mmd array");
}

static AIOFini(void)
{
        free(mmds);
        free(iods);
        free(events);
}

static void *DAOS_Create(char *testFileName, IOR_param_t *param)
{
        return DAOS_Open(testFileName, param);
}

static void *DAOS_Open(char *testFileName, IOR_param_t *param)
{
        struct fileDescriptor *fd;

        if (!initialized) {
                Init();
                initialized = 1;
        }

        fd = malloc(sizeof *fd);
        if (fd == NULL)
                ERR("Failed to allocate fd");

        /*
         * If param->open is not WRITE, the container must be created by a
         * "symmetrical" write session first.
         */
        ContainerOpen(testFileName, param, &fd->container, &fd->epoch);

        if (param->open == WRITE)
                /*
                 * All write operations form a single transaction.
                 */
                fd->epoch.seq++;

        ObjectOpen(fd->container, &fd->object, &fd->epoch, param);

        AIOInit(param);

        return fd;
}

static IOR_offset_t DAOS_Xfer(int access, void *file, IOR_size_t *buffer,
                              IOR_offset_t length, IOR_param_t *param)
{
        struct fileDescriptor *fd = file;
        struct daos_event      parent;
        struct daos_event     *event;
        daos_off_t             offset;
        daos_size_t            ioSize;
        int                    i;
        int                    rc;

        assert(nEvents == param->aiosPerTransfer);

        /*
         * This assumes "rankOffset == 0 && segmentCount == 1 && length %
         * aiosPerTransfer == 0".  There should be a "check" method in
         * ior_aiori_t.
         */
        assert(param->segmentCount == 1);
        assert(param->reorderTasks == 0);
        assert(param->reorderTasksRandom == 0);
        offset = param->offset - rank * param->blockSize;
        ioSize = length / param->aiosPerTransfer;

        rc = daos_event_init(&parent, eventQueue, NULL /* orphan */);
        DCHECK(rc, "Failed to initialize parent event");

        for (i = 0; i < nEvents; i++) {
                rc = daos_event_init(&events[i], eventQueue, &parent);
                DCHECK(rc, "Failed to initialize child event");
        }

        for (i = 0; i < param->aiosPerTransfer; i++) {
                iods[i].iod_nfrag = 1;
                /*
                 * This assumes "rankOffset == 0 && segmentCount == 1".
                 */
                iods[i].iod_frag[0].if_offset = offset + ioSize * i;
                iods[i].iod_frag[0].if_nob = ioSize;

                mmds[i].mmd_nfrag = 1;
                mmds[i].mmd_frag[0].mf_addr = (char *) buffer + ioSize * i;
                mmds[i].mmd_frag[0].mf_nob = ioSize;

                if (access == WRITE) {
                        rc = daos_object_write(fd->object, fd->epoch, &mmds[i],
                                               &iods[i], &events[i]);
                        DCHECK(rc, "Failed to start write operation");
                } else {
                        rc = daos_object_read(fd->object, fd->epoch, &mmds[i],
                                              &iods[i], &events[i]);
                        DCHECK(rc, "Failed to start read operation");
                }
        }

        rc = daos_eq_poll(eventQueue, 1 /* only if has inflight */,
                          DAOS_EQ_WAIT, 1 /* only 1 event pointer*/, &event);
        DCHECK(rc, "Failed to poll event queue");

        if (rc != 1)
                DCHECK(EINVAL, "Unexpected number of events: %d", rc);

        if (event != &parent)
                ERR("Unexpected event");

        DCHECK(event->ev_error, "Failed to transfer");

        for (i = 0; i < nEvents; i++)
                daos_event_fini(&events[i]);

        daos_event_fini(&parent);

        return length;
}

static void DAOS_Close(void *file, IOR_param_t *param)
{
        struct fileDescriptor *fd = file;
        int                    rc;

        AIOFini();

        ObjectClose(fd->object);

        if (param->open == WRITE) {
                MPI_CHECK(MPI_Barrier(param->testComm),
                          "Failed to synchronize processes");

                if (rank == 0) {
                        rc = daos_epoch_commit(fd->container, fd->epoch,
                                               1 /* sync */, NULL,
                                               NULL /* synchronous */);
                        DCHECK(rc, "Failed to commit object write");
                }
        }

        ContainerClose(fd->container, &fd->epoch, param);

        free(fd);

        if (initialized) {
                Fini();
                initialized = 0;
        }
}

static void DAOS_Delete(char *testFileName, IOR_param_t *param)
{
        int rc;

        if (!initialized) {
                Init();
                initialized = 1;
        }

        rc = daos_container_unlink(testFileName, NULL /* synchronous */);
        DCHECK(rc, "Failed to unlink container");
}

static void DAOS_SetVersion(IOR_param_t *test)
{
        strcpy(test->apiVersion, test->api);
}

static void DAOS_Fsync(void *file, IOR_param_t *param)
{
        /*
         * Inapplicable at the moment.
         */
}

static IOR_offset_t DAOS_GetFileSize(IOR_param_t *test, MPI_Comm testComm,
                                     char *testFileName)
{
        /*
         * Sizes are inapplicable to containers at the moment.
         */
        return 0;
}
