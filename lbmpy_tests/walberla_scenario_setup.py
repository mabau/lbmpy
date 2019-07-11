import waLBerla.field as field
from waLBerla import createUniformBlockGrid, createUniformBufferedScheme, makeSlice


def create_walberla_lattice_model(stencil, method, relaxation_rates, compressible=False, order=2,
                                  force_model='none', force=(0, 0, 0), **_):
    from waLBerla import lbm

    if method.lower() == 'srt':
        collision_model = lbm.collisionModels.SRT(relaxation_rates[0])
    elif method.lower() == 'trt':
        collision_model = lbm.collisionModels.TRT(relaxation_rates[0], relaxation_rates[1])
    elif method.lower() == 'mrt':
        if stencil != 'D3Q19':
            raise ValueError("MRT is available for D3Q19 only in walberla")
        collision_model = lbm.collisionModels.D3Q19MRT(*relaxation_rates[1:7])
    else:
        raise ValueError("Unknown method: " + str(method))

    if len(force) == 2:
        force = (force[0], force[1], 0)

    if force_model is None or force_model.lower() == 'none':
        force_model = lbm.forceModels.NoForce()
    elif force_model.lower() == 'simple':
        force_model = lbm.forceModels.SimpleConstant(force)
    elif force_model.lower() == 'luo':
        force_model = lbm.forceModels.LuoConstant(force)
    elif force_model.lower() == 'guo':
        force_model = lbm.forceModels.GuoConstant(force)
    else:
        raise ValueError("Unknown force model")
    return lbm.makeLatticeModel(stencil, collision_model, force_model, compressible, order)


def create_force_driven_channel_2d(force, radius, length, **kwargs):
    from waLBerla import lbm

    kwargs['force'] = tuple([force, 0, 0])

    domain_size = (length, 2 * radius, 1)

    lattice_model = create_walberla_lattice_model(**kwargs)
    blocks = createUniformBlockGrid(cells=domain_size, periodic=(1, 0, 1))

    # Adding fields
    lbm.addPdfFieldToStorage(blocks, "pdfs", lattice_model, velocityAdaptor="vel", densityAdaptor="rho",
                             initialDensity=1.0)
    field.addFlagFieldToStorage(blocks, 'flags')
    lbm.addBoundaryHandlingToStorage(blocks, 'boundary', 'pdfs', 'flags')

    # Communication
    communication = createUniformBufferedScheme(blocks, lattice_model.communicationStencilName)
    communication.addDataToCommunicate(field.createPackInfo(blocks, 'pdfs'))

    # Setting boundaries
    for block in blocks:
        b = block['boundary']
        if block.atDomainMaxBorder[1]:  # N
            b.forceBoundary('NoSlip', makeSlice[:, -1, :, 'g'])
        if block.atDomainMinBorder[1]:  # S
            b.forceBoundary('NoSlip', makeSlice[:, 0, :, 'g'])

        b.fillWithDomain()

    sweep = lbm.makeCellwiseSweep(blocks, "pdfs", flagFieldID='flags', flagList=['fluid']).streamCollide

    def time_loop(time_steps):
        for t in range(time_steps):
            communication()
            for B in blocks:
                B['boundary']()
            for B in blocks:
                sweep(B)
        full_pdf_field = field.toArray(field.gather(blocks, 'pdfs', makeSlice[:, :, :]), withGhostLayers=False)
        density = field.toArray(field.gather(blocks, 'rho', makeSlice[:, :, :]), withGhostLayers=False)
        velocity = field.toArray(field.gather(blocks, 'vel', makeSlice[:, :, :]), withGhostLayers=False)
        full_pdf_field = full_pdf_field[:, :, 0, :]
        density = density[:, :, 0, 0]
        velocity = velocity[:, :, 0, :2]
        return full_pdf_field, density, velocity

    return time_loop


def create_lid_driven_cavity(domain_size, lid_velocity=0.005, **kwargs):
    from waLBerla import lbm

    d = len(domain_size)

    if 'stencil' not in kwargs:
        kwargs['stencil'] = 'D2Q9' if d == 2 else 'D3Q27'

    if d == 2:
        domain_size = (domain_size[0], domain_size[1], 1)

    lattice_model = create_walberla_lattice_model(**kwargs)
    blocks = createUniformBlockGrid(cells=domain_size, periodic=(1, 1, 1))

    # Adding fields
    lbm.addPdfFieldToStorage(blocks, "pdfs", lattice_model, velocityAdaptor="vel", densityAdaptor="rho",
                             initialDensity=1.0)
    field.addFlagFieldToStorage(blocks, 'flags')
    lbm.addBoundaryHandlingToStorage(blocks, 'boundary', 'pdfs', 'flags')

    # Communication
    communication = createUniformBufferedScheme(blocks, lattice_model.communicationStencilName)
    communication.addDataToCommunicate(field.createPackInfo(blocks, 'pdfs'))

    # Setting boundaries
    for block in blocks:
        b = block['boundary']
        if block.atDomainMaxBorder[1]:  # N
            b.forceBoundary('UBB', makeSlice[:, -1, :, 'g'], {'x': lid_velocity})
        if block.atDomainMinBorder[1]:  # S
            b.forceBoundary('NoSlip', makeSlice[:, 0, :, 'g'])
        if block.atDomainMinBorder[0]:  # W
            b.forceBoundary('NoSlip', makeSlice[0, :, :, 'g'])
        if block.atDomainMaxBorder[0]:  # E
            b.forceBoundary('NoSlip', makeSlice[-1, :, :, 'g'])
        if block.atDomainMinBorder[2] and d == 3:  # T
            b.forceBoundary('NoSlip', makeSlice[:, :, 0, 'g'])
        if block.atDomainMaxBorder[2] and d == 3:  # B
            b.forceBoundary('NoSlip', makeSlice[:, :, -1, 'g'])

        b.fillWithDomain()

    sweep = lbm.makeCellwiseSweep(blocks, "pdfs", flagFieldID='flags', flagList=['fluid']).streamCollide

    def time_loop(time_steps):
        for t in range(time_steps):
            communication()
            for B in blocks:
                B['boundary']()
            for B in blocks:
                sweep(B)
        full_pdf_field = field.toArray(field.gather(blocks, 'pdfs', makeSlice[:, :, :]), withGhostLayers=False)
        density = field.toArray(field.gather(blocks, 'rho', makeSlice[:, :, :]), withGhostLayers=False)
        velocity = field.toArray(field.gather(blocks, 'vel', makeSlice[:, :, :]), withGhostLayers=False)
        if d == 2:
            full_pdf_field = full_pdf_field[:, :, 0, :]
            density = density[:, :, 0, 0]
            velocity = velocity[:, :, 0, :2]
        elif d == 3:
            density = density[:, :, :, 0]

        return full_pdf_field, density, velocity

    return time_loop
