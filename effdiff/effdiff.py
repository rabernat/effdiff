from pylab import *
from warnings import warn

# class for working with effective diffusivities
class EffDiffEngine:
    """Make is easier doing many effective diffusivity operations on the same
    grid."""

    def __init__(self, rac, dx, dy, N=None):
        
        self.rac = np.ma.masked_array(rac)
        self.dx = dx
        self.dy = dy

        (self.Ny, self.Nx) = self.rac.shape

        if (rac.shape != dx.shape) | (rac.shape != dy.shape):
            raise ValueError("Shapes of rac, dx, and dy are inconsistent.")

        if N == None:
            N = Ny
        self.N = N

        # calculate A as a function of y
        self.tot_area = rac.sum()
        self.A = linspace(0, self.tot_area, N)

    def calc_Le2(self, q, debug=False):
        if (q.shape != self.rac.shape):
            raise ValueError("q doesn't have the same shape as rac")
        return calc_Le2_atA(q, self.A, self.rac, self.dx, self.dy, debug=debug)


def grow_land(land_orig, N=1):
    """Grow the land mask defined by the array land by N points.
    """

    land = land_orig.copy()
    (Ny,Nx) = land.shape

    for n in range(N):
        gland = land.copy()
        for j in arange(Ny):
            for i in arange(Nx):
                # only do something if it is not land
                if (~land[j,i]):
                    do_switch = 0
                    # check left
                    if i>0:
                        do_switch += land[j,i-1]
                    # check right
                    if i<(Nx-1):
                        do_switch += land[j,i+1]
                    # check down
                    if j>0:
                        do_switch += land[j-1,i]
                    # check up
                    if j<(Ny-1):
                        do_switch += land[j+1,i]
                    # check left down
                    if ( i>0 ) & ( j>0 ):
                        do_switch += land[j-1,i-1]
                    # check right down
                    if ( i<(Nx-1) ) & ( j>0 ):
                        do_switch += land[j-1,i+1]
                    # check left up
                    if ( i>0 ) & ( j<(Ny-1) ):
                        do_switch += land[j+1,j+1]
                    # check right up
                    if ( i<(Nx-1) ) & ( j<(Ny-1) ):
                        do_switch += land[j+1,i+1]
                    
                    # change value if land has been found nearby
                    gland[j,i] = (do_switch > 0)
        land = gland

    return land


def calc_Le2_atA(c, A, rac, dx, dy, debug=False):
    """Calculate effective length (squared) of tracer contours.
    Following notation of Shuckburgh & Haynes 2003
     OUTPUTS:
      Le2: effective length^2 at A
      C: tracer value at area intervals
      Fd: diffusive flux integrated over the contour C(A)
      (note: this needs to be scaled by k, the small-scale diffusion)

     INPUTS:
      c: the two-dimensional tracer concentration field
      A: specified area intervals at which to do calculation
      c: tracer concentration matrix (possibly a masked array)
      dx: grid size (same size as c)
      dy: grid size (same size as c)
      rac: grid area (same size as c)

      L^2_eq = <|grad_c|^2>_A / (dC/dA)^2
    """

    # initialize ouputs
    result = dict(Le2=[], C=[], Fd=[])

    # check if c is a masked array
    if not isinstance(rac, ma.MaskedArray):
        warn('Area array (rac) is not a masked array; assuming no land.')
    else:
        rac = ma.masked_array(rac)

    # size of output array
    N = len(A)

    # size of linear tracer spacing array
    Nc = N

    # calculate area enclosed by tracer contours
    c_min = c.min()
    c_max = c.max()
    C_lin = linspace(c_min, c_max, Nc) # linearly spaced tracer contours
    
    # take gradients
    # second-order centered difference is used by numpy 
    grad_c = gradient(c)
    grad_c_y = grad_c[0] / dy
    grad_c_x = grad_c[1] / dx
    # we need to integrate |grad c|^2, this makes it easy
    grad_c2xRAC = (grad_c_x**2 + grad_c_y**2) * rac.filled(0.)
    
    # loop through each contour, find area, and integrate grad C2
    A_C_lin = zeros(Nc) # the area inside each contour
    grad_c2_C_lin = zeros(Nc) # integral of c^2 inside contour
    for n in range(Nc):
        idx = find( (c <= C_lin[n]) & ~rac.mask )
        A_C_lin[n] = rac.flatten()[idx].sum()
        grad_c2_C_lin[n] = grad_c2xRAC.flatten()[idx].sum()

    # interp to linear spacing in A
    C = interp(A, A_C_lin, C_lin)
    X = interp(A, A_C_lin, grad_c2_C_lin)

    dCdA = gradient(C) / gradient(A)
    dXdA = gradient(X) / gradient(A)

    Le2 = dXdA / (dCdA**2)

    if debug:
        print 'Minimum tracer: %.4e' % (c_min)
        print 'Maximum tracer: %.4e' % (c_max)
        print 'Minimum area A (given): %.4e' % (A.min())
        print 'Maximum area A (given): %.4e' % (A.max())
        print 'Minimum area A (found): %.4e' % (A_C_lin.min())
        print 'Maximum area A (found): %.4e' % (A_C_lin.max())
        print 'Minimum tracer (C): %.4e' % (C.min())
        print 'Maximum tracer (C): %.4e' % (C.max())
        figure(99); clf()
        plot(A, dXdA); title('dXdA');
        figure(999); clf()
        plot(A, dCdA); title('dCdA')
        draw(); show()

    result['C'] = C
    result['Le2'] = Le2
    return result

        
