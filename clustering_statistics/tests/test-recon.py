"""
salloc -N 1 -C "gpu&hbm80g" -t 00:30:00 --gpus 4 --qos interactive --account desi_g
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh new
srun -n 4 python test.py
"""
import os
import sys
import logging
import functools
from pathlib import Path

import jax
import numpy as np
import lsstypes as types

from clustering_statistics import tools, setup_logging, compute_stats_from_options
from clustering_statistics.recon_tools import compute_reconstruction
from clustering_statistics.tools import fill_fiducial_options, _merge_options, Catalog, setup_logging

logger = logging.getLogger('test_recon')

def _make_list_zrange(zranges):
    if np.ndim(zranges[0]) == 0:
        zranges = [zranges]
    return list(zranges)

def recon_output(get_catalog_fn=None, get_stats_fn=tools.get_stats_fn,
                                read_clustering_catalog=tools.read_clustering_catalog,
                                read_full_catalog=tools.read_full_catalog, analysis='full_shape', **kwargs):
    kwargs = fill_fiducial_options(kwargs, analysis=analysis)
    catalog_options = kwargs['catalog']
    tracers = list(catalog_options.keys())

    zranges = {tracer: _make_list_zrange(catalog_options[tracer]['zrange']) for tracer in tracers}

    if get_catalog_fn is not None:
        read_clustering_catalog = functools.partial(read_clustering_catalog, get_catalog_fn=get_catalog_fn)
        read_full_catalog = functools.partial(read_full_catalog, get_catalog_fn=get_catalog_fn)

    data, randoms = {}, {}
    for tracer in tracers:
        _catalog_options = dict(catalog_options[tracer])
        _catalog_options['zrange'] = (min(zrange[0] for zrange in zranges[tracer]), max(zrange[1] for zrange in zranges[tracer]))
        recon_options = kwargs['recon'][tracer]
        _catalog_options |= {key: recon_options.pop(key) for key in list(recon_options) if key in ['nran', 'zrange']}
        if any(name in catalog_options.get('weight', '') for name in ['bitwise', 'compntile']):
            # sets NTILE-MISSING-POWER (missing_power) and per-tile completeness (completeness)
            _catalog_options['binned_weight'] = read_full_catalog(kind='parent_data', **_catalog_options, attrs_only=True)

        data[tracer] = read_clustering_catalog(kind='data', **_catalog_options, concatenate=True)
        
        # pop as we don't need it anymore
        
        randoms[tracer] = read_clustering_catalog(kind='randoms', **_catalog_options, concatenate=False)

    from jaxpower import create_sharding_mesh
    with create_sharding_mesh() as sharding_mesh:

        data_rec, randoms_rec = {}, {}
        for tracer in tracers:
            recon_options = kwargs['recon'][tracer]
            # local sizes to select positions
            data[tracer]['POSITION_REC'], randoms_rec_positions = compute_reconstruction(lambda: (data[tracer], Catalog.concatenate(randoms[tracer])), **recon_options)
            start = 0
            for random in randoms[tracer]:
                size = len(random['POSITION'])
                random['POSITION_REC'] = randoms_rec_positions[start:start + size]
                start += size
            randoms[tracer] = randoms[tracer][:catalog_options[tracer]['nran']]  # keep only relevant random files
    return data, randoms

def test_recon_output():
    stats_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    stat = 'recon_mesh2_spectrum'
    for tracer in ['LRG']:
        zrange = tools.propose_fiducial('zranges', tracer)[0]
        for region in ['NGC', 'SGC'][:1]:
            catalog_options = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, imock=451, nran=2)
            catalog_options.update(expand={'parent_randoms_fn': tools.get_catalog_fn(kind='parent_randoms', version='data-dr2-v2', tracer=tracer, nran=catalog_options['nran'])})
            
            data, randoms = recon_output(catalog=catalog_options, get_stats_fn=functools.partial(tools.get_stats_fn, stats_dir=stats_dir), mesh2_spectrum={}, particle2_correlation={}, recon={'bias':2.0})

    # show some structure of data and randoms
        print(type(data[tracer]), type(randoms[tracer]), len(randoms[tracer]), type(randoms[tracer][0]))
        print(data[tracer].keys(), data[tracer]['POSITION_REC'].shape)
        print(randoms[tracer][0].keys(), randoms[tracer][0]['POSITION_REC'].shape)

        data[tracer].write(stats_dir / f'jax_recon_{tracer}_{region}_clustering.dat.h5')
        for i, random in enumerate(randoms[tracer]):
            random.write(stats_dir / f'jax_recon_{tracer}_{i}_{region}_clustering.ran.h5')
    
    return data, randoms

if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    jax.distributed.initialize()
    setup_logging()
    data, randoms = test_recon_output()


