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
#include <libgen.h>
#include <stdbool.h>
#include <daos/daos_api.h>

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

struct aio {
        struct daos_iod         a_iod;
        struct daos_io_frag     a_io_frag;
        struct daos_mmd         a_mmd;
        struct daos_mm_frag     a_mm_frag;
        struct daos_event       a_event;
};

static daos_handle_t      eventQueue;
static struct aio        *aios;
static int                nAios;
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

static char *
path_get_dir(const char *path)
{
        char *p;
        char *d;

        p = strdup(path);
        if (p == NULL)
                return NULL;
        d = strdup(dirname(p));
        free(p);
        return d;
}

static int
target_compare(const void *a, const void *b)
{
        return (*(const unsigned int *) a) - (*(const unsigned int *) b);
}

static void SysInfoInit(const char *path)
{
        daos_handle_t        sysContainer;
        struct daos_location loc;
        struct daos_loc_key *lks;
        unsigned int         lkn;
        int                  i;
        int                  rc;

        if (rank == 0) {
#ifdef HAVE_DAOS_POSIX
                path = getenv("DAOS_POSIX");
#endif
                rc = daos_sys_open(path, &sysContainer, NULL /* synchronous */);
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

                rc = daos_sys_close(sysContainer, NULL /* synchronous */);
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

                qsort(targets, nTargets, sizeof *targets, target_compare);
        }

        MPI_Bcast(targets, nTargets, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}

static void SysInfoFini(void)
{
        free(targets);
}

static void Init(void)
{
        int rc;

#ifdef HAVE_DAOS_POSIX
        rc = daos_posix_init();
        DCHECK(rc, "Failed to initialize daos-posix");
#endif

	rc = daos_eq_create(&eventQueue);
        DCHECK(rc, "Failed to create event queue");
}

static void Fini(void)
{
        int rc;

	rc = daos_eq_destroy(eventQueue);
        DCHECK(rc, "Failed to destroy event queue");

#ifdef HAVE_DAOS_POSIX
        daos_posix_finalize();
#endif
}

static void ShardAdd(daos_handle_t container, daos_epoch_t *epoch,
                     IOR_param_t *param)
{
        if (rank == 0) {
                unsigned int *s;
                unsigned int *t;
                int           i;
                int           rc;

                s = malloc((sizeof *s) * param->daos_n_shards);
                if (s == NULL)
                        ERR("Failed to allocate shard array");

                t = malloc((sizeof *t) * param->daos_n_shards);
                if (t == NULL)
                        ERR("Failed to allocate target array");

                for (i = 0; i < param->daos_n_shards; i++) {
                        s[i] = i;
                        t[i] = targets[i % param->daos_n_targets];
                }

                rc = daos_shard_add(container, *epoch, param->daos_n_shards,
                                    t, s, NULL /* synchronous */);
                DCHECK(rc, "Failed to create shards");

                free(t);
                free(s);
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

        if (param->open == WRITE) {
                epoch->seq++;

                ShardAdd(*container, epoch, param);

                if (rank == 0) {
                        rc = daos_epoch_commit(*container, *epoch, 1 /* sync */,
                                               NULL, NULL /* synchronous */);
                        DCHECK(rc, "Failed to commit shard creation");
                }
        }

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

        oid.o_id_hi = rank % param->daos_n_shards;
        oid.o_id_lo = rank / param->daos_n_shards;

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

static void AIOInit(IOR_param_t *param)
{
        nAios = param->aiosPerTransfer;

        aios = malloc((sizeof *aios) * nAios);
        if (aios == NULL)
                ERR("Failed to allocate aio array");
}

static void AIOFini(void)
{
        free(aios);
}

static void *DAOS_Create(char *testFileName, IOR_param_t *param)
{
        return DAOS_Open(testFileName, param);
}

static void *DAOS_Open(char *testFileName, IOR_param_t *param)
{
        struct fileDescriptor *fd;
        char                  *dir;

        if (!initialized) {
                Init();
                initialized = 1;
        }

        dir = path_get_dir(testFileName);
        if (dir == NULL)
                ERR("Failed to get path directory");

        SysInfoInit(dir);

        if (param->daos_n_targets == -1)
                param->daos_n_targets = nTargets;
        else if (param->daos_n_targets > nTargets)
                ERR("'daosntargets' must <= the number of all targets");

        if (param->daos_n_shards == -1)
                param->daos_n_shards = param->daos_n_targets;

        free(dir);

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

        assert(nAios == param->aiosPerTransfer);

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

        for (i = 0; i < nAios; i++) {
                rc = daos_event_init(&aios[i].a_event, eventQueue, &parent);
                DCHECK(rc, "Failed to initialize child event");
        }

        for (i = 0; i < param->aiosPerTransfer; i++) {
                aios[i].a_iod.iod_nfrag = 1;
                /*
                 * This assumes "rankOffset == 0 && segmentCount == 1".
                 */
                aios[i].a_iod.iod_frag[0].if_offset = offset + ioSize * i;
                aios[i].a_iod.iod_frag[0].if_nob = ioSize;

                aios[i].a_mmd.mmd_nfrag = 1;
                aios[i].a_mmd.mmd_frag[0].mf_addr = (char *) buffer +
                                                    ioSize * i;
                aios[i].a_mmd.mmd_frag[0].mf_nob = ioSize;

                if (access == WRITE) {
                        rc = daos_object_write(fd->object, fd->epoch,
                                               &aios[i].a_mmd, &aios[i].a_iod,
                                               &aios[i].a_event);
                        DCHECK(rc, "Failed to start write operation");
                } else {
                        rc = daos_object_read(fd->object, fd->epoch,
                                              &aios[i].a_mmd, &aios[i].a_iod,
                                              &aios[i].a_event);
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

        for (i = 0; i < nAios; i++)
                daos_event_fini(&aios[i].a_event);

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

        SysInfoFini();

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
