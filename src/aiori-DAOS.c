/*
 * -*- mode: c; c-basic-offset: 8; indent-tabs-mode: nil; -*-
 * vim:expandtab:shiftwidth=8:tabstop=8:
 */
/*
 * SPECIAL LICENSE RIGHTS-OPEN SOURCE SOFTWARE
 * The Government's rights to use, modify, reproduce, release, perform, display,
 * or disclose this software are subject to the terms of Contract No. B599860,
 * and the terms of the GNU General Public License version 2.
 * Any reproduction of computer software, computer software documentation, or
 * portions thereof marked with this legend must also reproduce the markings.
 */
/*
 * Copyright (c) 2013, 2016 Intel Corporation.
 */
/*
 * This file implements the abstract I/O interface for DAOS.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdint.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <libgen.h>
#include <stdbool.h>
#include <daos_sr.h>

#include "ior.h"
#include "aiori.h"
#include "iordef.h"
#include "list.h"

/**************************** P R O T O T Y P E S *****************************/

static void DAOS_Init(IOR_param_t *);
static void DAOS_Fini(IOR_param_t *);
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
        DAOS_GetFileSize,
        DAOS_Init,
        DAOS_Fini
};

enum handleType {
        POOL_HANDLE,
        CONTAINER_HANDLE
};

struct fileDescriptor {
        daos_handle_t  container;
        daos_co_info_t containerInfo;
        daos_handle_t  object;
        daos_epoch_t   epoch;
};

struct aio {
        cfs_list_t              a_list;
        char                    a_dkeyBuf[32];
        daos_dkey_t             a_dkey;
        daos_recx_t             a_recx;
        unsigned char           a_csumBuf[32];
        daos_csum_buf_t         a_csum;
        daos_epoch_range_t      a_epochRange;
        daos_vec_iod_t          a_iod;
        daos_iov_t              a_iov;
        daos_sg_list_t          a_sgl;
        struct daos_event       a_event;
};

static daos_handle_t       eventQueue;
static struct daos_event **events;
static unsigned char      *buffers;
static int                 nAios;
static daos_handle_t       pool = DAOS_HDL_INVAL;
static daos_pool_info_t    poolInfo;
static daos_oclass_id_t    objectClass = 1;

static CFS_LIST_HEAD(aios);

/***************************** F U N C T I O N S ******************************/

#define DCHECK(rc, format, ...)                                         \
do {                                                                    \
        int _rc = (rc);                                                 \
                                                                        \
        if (_rc < 0) {                                                  \
                fprintf(stdout, "ior ERROR (%s:%d): %d: %d: "           \
                        format"\n", __FILE__, __LINE__, rank, _rc,      \
                        ##__VA_ARGS__);                                 \
                fflush(stdout);                                         \
                MPI_Abort(MPI_COMM_WORLD, -1);                          \
        }                                                               \
} while (0)

#define INFO(level, param, format, ...)                                 \
do {                                                                    \
        if (param->verbose >= level)                                    \
                printf("[%d] "format"\n", rank, ##__VA_ARGS__);          \
} while (0)

/* Distribute process 0's pool or container handle to others. */
static void HandleDistribute(daos_handle_t *handle, enum handleType type,
                             IOR_param_t *param)
{
        daos_iov_t global;
        int        rc;

        assert(type == POOL_HANDLE || !daos_handle_is_inval(pool));

        global.iov_buf = NULL;
        global.iov_buf_len = 0;
        global.iov_len = 0;

        if (rank == 0) {
                /* Get the global handle size. */
                if (type == POOL_HANDLE)
                        rc = dsr_pool_local2global(*handle, &global);
                else
                        rc = dsr_co_local2global(*handle, &global);
                DCHECK(rc, "Failed to get global handle size");
        }

        MPI_CHECK(MPI_Bcast(&global.iov_buf_len, 1, MPI_UINT64_T, 0,
                            param->testComm),
                  "Failed to bcast global handle buffer size");

        global.iov_buf = malloc(global.iov_buf_len);
        if (global.iov_buf == NULL)
                ERR("Failed to allocate global handle buffer");

        if (rank == 0) {
                if (type == POOL_HANDLE)
                        rc = dsr_pool_local2global(*handle, &global);
                else
                        rc = dsr_co_local2global(*handle, &global);
                DCHECK(rc, "Failed to create global handle");
        }

        MPI_CHECK(MPI_Bcast(global.iov_buf, global.iov_buf_len, MPI_BYTE, 0,
                            param->testComm),
                  "Failed to bcast global pool handle");

        if (rank != 0) {
                /* A larger-than-actual length works just fine. */
                global.iov_len = global.iov_buf_len;

                if (type == POOL_HANDLE)
                        rc = dsr_pool_global2local(global, handle);
                else
                        rc = dsr_co_global2local(pool, global, handle);
                DCHECK(rc, "Failed to get local handle");
        }

        free(global.iov_buf);
}

static void ContainerOpen(char *testFileName, IOR_param_t *param,
                          daos_handle_t *container, daos_co_info_t *info)
{
        int rc;

        if (rank == 0) {
                uuid_t       uuid;
                unsigned int dFlags;

                rc = uuid_parse(testFileName, uuid);
                DCHECK(rc, "Failed to parse 'testFile': %s", testFileName);

                if (param->open == WRITE &&
                    param->useExistingTestFile == FALSE) {
                        INFO(VERBOSE_2, param, "Creating container %s\n",
                             testFileName);

                        rc = dsr_co_create(pool, uuid, NULL /* ev */);
                        DCHECK(rc, "Failed to create container %s",
                               testFileName);
                }

                INFO(VERBOSE_2, param, "Openning container %s\n", testFileName);

                if (param->open == WRITE)
                        dFlags = DAOS_COO_RW;
                else
                        dFlags = DAOS_COO_RO;

                rc = dsr_co_open(pool, uuid, dFlags, NULL /* failed */,
                                 container, info, NULL /* ev */);
                DCHECK(rc, "Failed to open container %s", testFileName);

#if 0
                if (param->open != WRITE && param->daosWait != 0) {
                        daos_epoch_t e;

                        e = param->daosWait;

                        INFO(VERBOSE_2, param, "Waiting for epoch %lu\n", e);

                        rc = daos_epoch_wait(*container, &e,
                                             NULL /* ignore HLE */,
                                             NULL /* synchronous */);
                        DCHECK(rc, "Failed to wait for epoch %lu",
                               param->daosWait);
                }
#endif

                INFO(VERBOSE_2, param, "Container epoch state: \n");
                INFO(VERBOSE_2, param, "   HCE: %lu\n",
                     info->ci_epoch_state.es_hce);
                INFO(VERBOSE_2, param, "   LRE: %lu\n",
                     info->ci_epoch_state.es_lre);
                INFO(VERBOSE_2, param, "   LHE: %lu (%lx)\n",
                     info->ci_epoch_state.es_lhe, info->ci_epoch_state.es_lhe);
                INFO(VERBOSE_2, param, "  GHCE: %lu\n",
                     info->ci_epoch_state.es_glb_hce);
                INFO(VERBOSE_2, param, "  GLRE: %lu\n",
                     info->ci_epoch_state.es_glb_lre);
                INFO(VERBOSE_2, param, "  GLHE: %lu\n",
                     info->ci_epoch_state.es_glb_hpce);
        }

        HandleDistribute(container, CONTAINER_HANDLE, param);

        MPI_CHECK(MPI_Bcast(info, sizeof *info, MPI_BYTE, 0, param->testComm),
                  "Failed to broadcast container info");
}

static void ContainerClose(daos_handle_t container, IOR_param_t *param)
{
        int rc;

        if (rank != 0) {
                rc = dsr_co_close(container, NULL /* ev */);
                DCHECK(rc, "Failed to close container");
        }

        /* An MPI_Gather() call is probably more appropriate. */
        MPI_CHECK(MPI_Barrier(param->testComm),
                  "Failed to synchronize processes");

        if (rank == 0) {
                rc = dsr_co_close(container, NULL /* ev */);
                DCHECK(rc, "Failed to close container");
        }
}

static void ObjectOpen(daos_handle_t container, daos_handle_t *object,
                       daos_epoch_t epoch, IOR_param_t *param)
{
        daos_obj_id_t oid;
        unsigned int  flags;
        int           rc;

        oid.hi = 0;
        oid.mid = 0;
        oid.lo = 1;
        dsr_objid_generate(&oid, objectClass);

        if (rank == 0) {
#if 0
                daos_oclass_attr_t attr = {
                        .ca_schema              = DAOS_OS_STRIPED,
                        .ca_resil_degree        = 0,
                        .ca_resil               = DAOS_RES_REPL,
                        .ca_nstripes            = 4,
                        .u.repl                 = {
                                .r_method       = 0,
                                .r_num          = 2
                        }
                };

                rc = dsr_oclass_register(container, objectClass, &attr,
                                         NULL /* ev */);
                DCHECK(rc, "Failed to register object class");
#endif

                rc = dsr_obj_declare(container, oid, epoch, NULL /* oa */,
                                     NULL /* ev */);
                DCHECK(rc, "Failed to declare object");
        }

        if (param->open == WRITE)
                flags = DAOS_OO_RW;
        else
                flags = DAOS_OO_RO;

        rc = dsr_obj_open(container, oid, epoch, flags, object, NULL /* ev */);
        DCHECK(rc, "Failed to open object");
}

static void ObjectClose(daos_handle_t object)
{
        int rc;

        rc = dsr_obj_close(object, NULL /* ev */);
        DCHECK(rc, "Failed to close object");
}

static void AIOInit(IOR_param_t *param)
{
        struct aio *aio;
        int         i;
        int         rc;

        rc = posix_memalign((void **) &buffers, sysconf(_SC_PAGESIZE),
                            param->transferSize * param->daosAios);
        DCHECK(rc, "Failed to allocate buffer array");

        for (i = 0; i < param->daosAios; i++) {
                aio = malloc(sizeof *aio);
                if (aio == NULL)
                        ERR("Failed to allocate aio array");

                memset(aio, 0, sizeof *aio);

                aio->a_dkey.iov_buf = aio->a_dkeyBuf;
                aio->a_dkey.iov_buf_len = sizeof aio->a_dkeyBuf;

                aio->a_recx.rx_rsize = param->transferSize;
                aio->a_recx.rx_nr = 1;

                aio->a_csum.cs_csum = &aio->a_csumBuf;
                aio->a_csum.cs_buf_len = sizeof aio->a_csumBuf;
                aio->a_csum.cs_len = aio->a_csum.cs_buf_len;

                aio->a_epochRange.epr_hi = DAOS_EPOCH_MAX;

                aio->a_iod.vd_name.iov_buf = "data";
                aio->a_iod.vd_name.iov_buf_len =
                        strlen(aio->a_iod.vd_name.iov_buf) + 1;
                aio->a_iod.vd_name.iov_len = aio->a_iod.vd_name.iov_buf_len;
                aio->a_iod.vd_nr = 1;
                aio->a_iod.vd_recxs = &aio->a_recx;
                aio->a_iod.vd_csums = &aio->a_csum;
                aio->a_iod.vd_eprs = &aio->a_epochRange;

                aio->a_iov.iov_buf = buffers + param->transferSize * i;
                aio->a_iov.iov_buf_len = param->transferSize;
                aio->a_iov.iov_len = aio->a_iov.iov_buf_len;

                aio->a_sgl.sg_nr.num = 1;
                aio->a_sgl.sg_iovs = &aio->a_iov;

                rc = daos_event_init(&aio->a_event, eventQueue,
                                     NULL /* parent */);
                DCHECK(rc, "Failed to initialize event for aio[%d]", i);

                cfs_list_add(&aio->a_list, &aios);

                INFO(VERBOSE_3, param, "Allocated AIO %p: buffer %p\n", aio,
                     aio->a_iov.iov_buf);
        }

        nAios = param->daosAios;

        events = malloc((sizeof *events) * param->daosAios);
        if (events == NULL)
                ERR("Failed to allocate events array");
}

static void AIOFini(IOR_param_t *param)
{
        struct aio *aio;
        struct aio *tmp;

        free(events);

        cfs_list_for_each_entry_safe(aio, tmp, &aios, a_list) {
                INFO(VERBOSE_3, param, "Freeing AIO %p: buffer %p\n", aio,
                     aio->a_iov.iov_buf);
                cfs_list_del_init(&aio->a_list);
                daos_event_fini(&aio->a_event);
                free(aio);
        }

        free(buffers);
}

static void AIOWait(IOR_param_t *param)
{
        struct aio *aio;
        int         i;
        int         rc;

        rc = daos_eq_poll(eventQueue, 0, DAOS_EQ_WAIT, param->daosAios,
                          events);
        DCHECK(rc, "Failed to poll event queue");
        assert(rc <= param->daosAios - nAios);

        for (i = 0; i < rc; i++) {
                aio = (struct aio *)
                      ((char *) events[i] -
                       (char *) (&((struct aio *) 0)->a_event));

                DCHECK(aio->a_event.ev_error, "Failed to transfer (%lu, %lu)",
                       aio->a_iod.vd_recxs->rx_idx,
                       aio->a_iod.vd_recxs->rx_nr);

                cfs_list_move(&aio->a_list, &aios);
                nAios++;

                if (param->verbose >= VERBOSE_3)
                INFO(VERBOSE_3, param, "Completed AIO %p: buffer %p\n", aio,
                     aio->a_iov.iov_buf);
        }

        INFO(VERBOSE_3, param, "Found %d completed AIOs (%d free %d busy)\n",
             rc, nAios, param->daosAios - nAios);
}

static void DAOS_Init(IOR_param_t *param)
{
        int rc;

        if (param->filePerProc)
                ERR("'filePerProc' not yet supported");
        if (param->daosStripeSize % param->transferSize != 0)
                ERR("'daosStripeSize' must be a multiple of 'transferSize'");
        if (param->transferSize % param->daosRecordSize != 0)
                ERR("'transferSize' must be a multiple of 'daosRecordSize'");

        rc = dsr_init();
        DCHECK(rc, "Failed to initialize daos");

        rc = daos_eq_create(&eventQueue);
        DCHECK(rc, "Failed to create event queue");

        if (rank == 0) {
                uuid_t           uuid;
                daos_rank_t      rank = 0;
                daos_rank_list_t ranks;

                if (strlen(param->daosPool) == 0)
                        ERR("'daosPool' must be specified");

                INFO(VERBOSE_2, param, "Connecting to pool %s\n",
                     param->daosPool);

                rc = uuid_parse(param->daosPool, uuid);
                DCHECK(rc, "Failed to parse 'daosPool': %s", param->daosPool);
                ranks.rl_nr.num = 1;
                ranks.rl_nr.num_out = 0;
                ranks.rl_ranks = &rank;

                rc = dsr_pool_connect(uuid, NULL /* grp */, &ranks,
                                      DAOS_PC_EX, NULL /* failed */, &pool,
                                      &poolInfo, NULL /* ev */);
                DCHECK(rc, "Failed to connect to pool %s", param->daosPool);
        }

        HandleDistribute(&pool, POOL_HANDLE, param);

        MPI_CHECK(MPI_Bcast(&poolInfo, sizeof poolInfo, MPI_BYTE, 0,
                            param->testComm),
                  "Failed to bcast pool info");

        if (param->daosStripeCount == -1)
                param->daosStripeCount = poolInfo.pi_ntargets * 64UL;
}

static void DAOS_Fini(IOR_param_t *param)
{
        int rc;

        rc = dsr_pool_disconnect(pool, NULL /* ev */);
        DCHECK(rc, "Failed to disconnect from pool %s", param->daosPool);

        rc = daos_eq_destroy(eventQueue, 0 /* flags */);
        DCHECK(rc, "Failed to destroy event queue");

        rc = dsr_fini();
        DCHECK(rc, "Failed to finalize daos");
}

static void *DAOS_Create(char *testFileName, IOR_param_t *param)
{
        return DAOS_Open(testFileName, param);
}

static void *DAOS_Open(char *testFileName, IOR_param_t *param)
{
        struct fileDescriptor *fd;
        daos_epoch_t           ghce;

        fd = malloc(sizeof *fd);
        if (fd == NULL)
                ERR("Failed to allocate fd");

        ContainerOpen(testFileName, param, &fd->container, &fd->containerInfo);

        ghce = fd->containerInfo.ci_epoch_state.es_glb_hce;
        if (param->open == WRITE) {
                int rc;

                if (param->daosEpoch == 0)
                        fd->epoch = ghce + 1;
                else if (param->daosEpoch <= ghce)
                        ERR("Can't modify committed epoch\n");
                else
                        fd->epoch = param->daosEpoch;

                if (rank == 0) {
                        daos_epoch_t e = fd->epoch;

                        rc = dsr_epoch_hold(fd->container, &fd->epoch,
                                            NULL /* state */, NULL /* ev */);
                        DCHECK(rc, "Failed to hold epoch");
                        assert(fd->epoch == e);
                }
        } else {
                if (param->daosEpoch == 0) {
                        if (param->daosWait == 0)
                                fd->epoch = ghce;
                        else
                                fd->epoch = param->daosWait;
                } else if (param->daosEpoch > ghce) {
                        ERR("Can't read uncommitted epoch\n");
                } else {
                        fd->epoch = param->daosEpoch;
                }
        }

        if (rank == 0)
                INFO(VERBOSE_2, param, "Accessing epoch %lu\n", fd->epoch);

        ObjectOpen(fd->container, &fd->object, fd->epoch, param);

        AIOInit(param);

        return fd;
}

static IOR_offset_t DAOS_Xfer(int access, void *file, IOR_size_t *buffer,
                              IOR_offset_t length, IOR_param_t *param)
{
        struct fileDescriptor *fd = file;
        struct aio            *aio;
        daos_off_t             offset;
        uint64_t               stripe;
        int                    rc;

        assert(length == param->transferSize);
        assert(param->offset % length == 0);

        /*
         * Find an available AIO descriptor.  If none, wait for one.
         */
        while (nAios == 0)
                AIOWait(param);
        aio = cfs_list_entry(aios.next, struct aio, a_list);
        cfs_list_move_tail(&aio->a_list, &aios);
        nAios--;

        stripe = (param->offset / param->daosStripeSize) %
                 param->daosStripeCount;
        rc = snprintf(aio->a_dkeyBuf, sizeof aio->a_dkeyBuf, "%lu", stripe);
        assert(rc < sizeof aio->a_dkeyBuf);
        aio->a_dkey.iov_len = strlen(aio->a_dkeyBuf) + 1;
        aio->a_recx.rx_idx = param->offset / param->daosRecordSize;
        aio->a_epochRange.epr_lo = fd->epoch;

        /*
         * If the data written will be checked later, we have to copy in valid
         * data instead of writing random bytes.  If the data being read is for
         * checking purposes, poison the buffer first.
         */
        if (access == WRITE && param->checkWrite)
                memcpy(aio->a_iov.iov_buf, buffer, length);
        else if (access == WRITECHECK || access == READCHECK)
                memset(aio->a_iov.iov_buf, '#', length);

        INFO(VERBOSE_3, param, "Starting AIO %p (%d free %d busy): access %d "
             "dkey '%s' iod <%llu, %llu> sgl <%p, %lu>\n", aio, nAios,
             param->daosAios - nAios, access, (char *) aio->a_dkey.iov_buf,
             (unsigned long long) aio->a_iod.vd_recxs->rx_idx,
             (unsigned long long) aio->a_iod.vd_recxs->rx_nr,
             aio->a_sgl.sg_iovs->iov_buf,
             (unsigned long long) aio->a_sgl.sg_iovs->iov_buf_len);

        if (access == WRITE) {
                rc = dsr_obj_update(fd->object, fd->epoch, &aio->a_dkey,
                                    1 /* nr */, &aio->a_iod, &aio->a_sgl,
                                    &aio->a_event);
                DCHECK(rc, "Failed to start update operation");
        } else {
                rc = dsr_obj_fetch(fd->object, fd->epoch, &aio->a_dkey,
                                   1 /* nr */, &aio->a_iod, &aio->a_sgl,
                                   NULL /* maps */, &aio->a_event);
                DCHECK(rc, "Failed to start fetch operation");
        }

        /*
         * If this is a WRITECHECK or READCHECK, we are expected to fill data
         * into the buffer before returning.  Note that if this is a READ, we
         * don't have to return valid data as WriteOrRead() doesn't care.
         */
        if (access == WRITECHECK || access == READCHECK) {
                while (param->daosAios - nAios > 0)
                        AIOWait(param);
                memcpy(buffer, aio->a_sgl.sg_iovs->iov_buf, length);
        }

        return length;
}

static void DAOS_Close(void *file, IOR_param_t *param)
{
        struct fileDescriptor *fd = file;
        int                    rc;

        while (param->daosAios - nAios > 0)
                AIOWait(param);
        AIOFini(param);

        ObjectClose(fd->object);

        if (param->open == WRITE && !param->daosWriteOnly) {
                /* Wait for everybody for to complete the writes. */
                MPI_CHECK(MPI_Barrier(param->testComm),
                          "Failed to synchronize processes");

                if (rank == 0) {
                        INFO(VERBOSE_2, param, "Flushing epoch %lu\n",
                             fd->epoch);

                        rc = dsr_epoch_flush(fd->container, fd->epoch,
                                             NULL /* state */, NULL /* ev */);
                        DCHECK(rc, "Failed to flush epoch");

                        INFO(VERBOSE_2, param, "Committing epoch %lu\n",
                             fd->epoch);

                        rc = dsr_epoch_commit(fd->container, fd->epoch,
                                              NULL /* state */, NULL /* ev */);
                        DCHECK(rc, "Failed to commit object write");
                }
        }

        ContainerClose(fd->container, param);

        free(fd);
}

static void DAOS_Delete(char *testFileName, IOR_param_t *param)
{
        uuid_t uuid;
        int    rc;

        rc = uuid_parse(testFileName, uuid);
        DCHECK(rc, "Failed to parse 'testFile': %s", testFileName);

        rc = dsr_co_destroy(pool, uuid, 0 /* !force */, NULL /* ev */);
        if (rc != -DER_NONEXIST)
                DCHECK(rc, "Failed to destroy container %s", testFileName);
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
