
import xubik0 as xu
import nifty8 as ift
import numpy as np
import timeit

from jax import config
config.update('jax_enable_x64', True)

ift.set_nthreads(8)

def get_kernels_and_sources(domain, psf_func):
    rnds = np.zeros(domain.shape)
    rnds[::20, ::20] = 1.
    cc = (np.arange(ss)*dd for ss,dd in zip(domain.shape, domain.distances))
    cc = np.meshgrid(*cc, indexing='ij')
    cx, cy = cc[0], cc[1]
    inds = np.where(rnds != 0.)
    cx = cx[inds]
    cy = cy[inds]

    res = np.zeros_like(rnds)
    for ix, iy, xx, yy in zip(inds[0],inds[1], cx, cy):
        psf = psf_func(xx, yy) * domain.scalar_dvol
        psf = np.roll(psf, ix, axis = 0)
        psf = np.roll(psf, iy, axis = 1)
        res += psf
    rnds = ift.makeField(domain, rnds)
    return ift.makeField(domain, res), rnds

# dirs
dir_path = "data/psf_info/"
fname = ["tm1_2dpsf_190219v05.fits", "tm1_2dpsf_190220v03.fits"]
filename = dir_path + fname[0]

# PSF Object
obs = xu.eROSITA_PSF(filename)

# more numbers about the observation
energy = '3000'
pointing_center = (1800, 1800)
fov = (3600, 3600)
npix = (512, 512)
dists = tuple(ff/pp for ff, pp in zip(fov, npix))
domain = ift.RGSpace(npix, distances=dists)

# psf_func = obs.psf_func_on_domain(energy, pointing_center, domain)
# kernels, sources = get_kernels_and_sources(domain, psf_func)

c2params = {'npatch': 8, 'margfrac': 0.062, 'want_cut': False}

test_psf = xu.build_erosita_psf(1, filename, energy, pointing_center, domain,
                                c2params["npatch"], c2params["margfrac"], c2params["want_cut"])
op1 = obs.make_psf_op(energy, pointing_center, domain,
                      conv_method='LINJAX', conv_params=c2params)
op2 = obs.make_psf_op(energy, pointing_center, domain,
                      conv_method='LIN', conv_params=c2params)

rnds = ift.from_random(op2.domain)

res1 = op1(rnds.val)
res2 = op2(rnds).val

print("")
print("Equality of LIN and LINJAX: ", np.allclose(res1, res2))

print('JIT LINJAX-PSF...')
t0 = timeit.default_timer()
res = op1(rnds)
t1 = timeit.default_timer()
print('...done JIT LIN-PSF')
print('Compile time MSC:', t1-t0)
res2 = op2(rnds)

t0 = timeit.default_timer()
res = op(rnds)
t1 = timeit.default_timer()
print('MSC:', t1-t0)
t0 = timeit.default_timer()
res2 = op2(rnds)
t1 = timeit.default_timer()
print('LIN:', t1-t0)

pl = ift.Plot()
pl.add(res, title = 'MSC')
pl.add(res2, title = 'LIN')
pl.add((res-res2).abs(), title='ABS Diff')
pl.output(nx=2, ny=2, xsize=16, ysize=16)

res = op(sources)
res2 = op2(sources)

pl = ift.Plot()
pl.add(res, title = 'MSC')
pl.add((res-kernels).abs(), title='Abs diff MSC')
pl.add(kernels, title = 'GT')

pl.add(res2, title = 'LIN')
pl.add((res2-kernels).abs(), title='Abs diff LIN')
pl.add(kernels, title = 'GT')
pl.output(nx=3, ny=2, xsize=16, ysize=10)
