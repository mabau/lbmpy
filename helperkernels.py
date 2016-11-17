

def initializePdfField(latticeModel, pdfArray):
    if latticeModel.compressible is None:
        pdfArray.fill(0.0)
    else:
        if latticeModel.dim == 2:
            for i, weight in enumerate(latticeModel.weights):
                pdfArray[:, :, i] = float(weight)
        elif latticeModel.dim == 3:
            for i, weight in enumerate(latticeModel.weights):
                pdfArray[:, :, :, i] = float(weight)
        else:
            raise NotImplementedError()



