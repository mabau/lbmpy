# Workaround for cython bug
# see https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
WORKAROUND = "Something"

import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simplex_projection_2d(object[double, ndim=3] c):
    cdef int xs, ys, num_phases, x, y, a, b, local_phases
    cdef unsigned int handled_mask = 0
    cdef double threshold = 1e-18
    cdef double phase_sum

    xs, ys, num_phases = c.shape

    for y in range(ys):
        for x in range(xs):
            local_phases = num_phases

            ## Mark zero phases
            for a in range(num_phases):
                if  -threshold < c[x, y, a] < threshold:
                    local_phases -= 1
                    handled_mask |= (1 << a)
                    c[x, y, a] = 0

            # Distribute negative phases to others
            a = 0
            while a < num_phases:
                if c[x, y, a] < 0.0:
                    handled_mask |= (1 << a)
                    local_phases -= 1
                    for b in range(num_phases): # distribute to unhandled phases
                        if handled_mask & (1 << b) == 0:
                            c[x, y, b] += c[x, y, a] / local_phases
                    c[x, y, a] = 0.0
                    a = -1  # restart loop, since other phases might have become negative
                a += 1

            # Normalize phases
            phase_sum = 0.0
            for a in range(num_phases):
                phase_sum += c[x, y, a]

            for a in range(num_phases):
                c[x, y, a] /= phase_sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simplex_projection_3d(object[double, ndim=4] c):
    cdef int xs, ys, num_phases, x, y, z, a, b, local_phases
    cdef unsigned int handled_mask = 0
    cdef double threshold = 1e-18
    cdef double phase_sum

    xs, ys, zs, num_phases = c.shape

    for z in range(zs):
        for y in range(ys):
            for x in range(xs):
                local_phases = num_phases

                ## Mark zero phases
                for a in range(num_phases):
                    if  -threshold < c[x, y, z, a] < threshold:
                        local_phases -= 1
                        handled_mask |= (1 << a)
                        c[x, y, z, a] = 0

                # Distribute negative phases to others
                a = 0
                while a < num_phases:
                    if c[x, y, z, a] < 0.0:
                        handled_mask |= (1 << a)
                        local_phases -= 1
                        for b in range(num_phases): # distribute to unhandled phases
                            if handled_mask & (1 << b) == 0:
                                c[x, y, z, b] += c[x, y, z, a] / local_phases
                        c[x, y, z, a] = 0.0
                        a = -1  # restart loop, since other phases might have become negative
                    a += 1

                # Normalize phases
                phase_sum = 0.0
                for a in range(num_phases):
                    phase_sum += c[x, y, z, a]

                for a in range(num_phases):
                    c[x, y, z, a] /= phase_sum
