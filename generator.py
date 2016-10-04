import cgen as c
import numpy as np
import sympy as sp
from sympy.core.cache import cacheit
from sympy.utilities.codegen import CCodePrinter
from sympy.tensor import IndexedBase
from collections import defaultdict
import warnings


COORDINATE_LOOP_COUNTER_NAME = "ctr"
FIELD_PREFIX = "field_"
BASE_PTR_PREFIX = FIELD_PREFIX + "base_"
FIELD_PTR_PREFIX = FIELD_PREFIX + "data_"
FIELD_SHAPE_PREFIX = FIELD_PREFIX + "shape_"
FIELD_STRIDE_PREFIX = FIELD_PREFIX + "stride_"

STRIDE_DTYPE = "const int *"
SHAPE_DTYPE = "const int *"


# --------------------------------------- Helper Functions -------------------------------------------------------------


class CodePrinter(CCodePrinter):
    def _print_Pow(self, expr):
        if expr.exp.is_integer and expr.exp.is_number:
            return '(' + '*'.join([self._print(expr.base)] * expr.exp) + ')'
        else:
            return super(SympyAssignment.CodePrinter, self)._print_Pow(expr)

codePrinter = CodePrinter()


def offsetToDirectionString(offsetTuple):
    nameComponents = (('W', 'E'),  # west, east
                      ('S', 'N'),  # south, north
                      ('B', 'T'),  # bottom, top
                      )
    names = ["", "", ""]
    for i in range(len(offsetTuple)):
        if offsetTuple[i] < 0:
            names[i] = nameComponents[i][0]
        elif offsetTuple[i] > 0:
            names[i] = nameComponents[i][1]
        if abs(offsetTuple[i]) > 1:
            names[i] = str(abs(offsetTuple[i])) + names[i]
    name = "".join(reversed(names))
    if name == "":
        name = "C"
    return name


def directionStringToOffset(directionStr, dim=3):
    offsetMap = {
        'C': np.array([0, 0, 0]),

        'W': np.array([-1, 0, 0]),
        'E': np.array([ 1, 0, 0]),

        'S': np.array([0, -1, 0]),
        'N': np.array([0,  1, 0]),

        'B': np.array([0, 0, -1]),
        'T': np.array([0, 0,  1]),
    }
    offset = np.array([0, 0, 0])

    while len(directionStr) > 0:
        factor = 1
        firstNonDigit = 0
        while directionStr[firstNonDigit].isdigit():
            firstNonDigit += 1
        if firstNonDigit > 0:
            factor = int(directionStr[:firstNonDigit])
            directionStr = directionStr[firstNonDigit:]
        curOffset = offsetMap[directionStr[0]]
        offset += factor * curOffset
        directionStr = directionStr[1:]
    return offset[:dim]


def fieldStruct(baseType=np.float64):
    return c.Struct("ndarray", [
        c.Pointer(c.POD(baseType, "data")),
        c.POD(np.long, "dim"),
        c.Pointer(c.POD(np.long, "strides")),
        c.Pointer(c.POD(np.long, "shape")),
    ])


def getLayoutFromNumpyField(npField):
    coordinates = list(range(len(npField.shape)))
    return [x for (y, x) in sorted(zip(npField.strides, coordinates), key=lambda pair: pair[0], reverse=True)]


def numpyDataTypeToString(dtype):
    if dtype == np.float64:
        return "double"
    elif dtype == np.float32:
        return "float"
    raise NotImplementedError()


class MyPOD(c.Declarator):
    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name

    def get_decl_pair(self):
        return [self.dtype], self.name


# --------------------------------------- AST Nodes  -------------------------------------------------------------------


class Node:
    def __init__(self, parent=None):
        self.parent = parent

    def atoms(self, argType):
        result = set()
        for arg in self.args:
            if type(arg) == argType:
                result.add(arg)
            result.update(arg.atoms(argType))
        return result


class KernelFunction(Node):

    class Argument:
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype
            self.isFieldPtrArgument = False
            self.isFieldShapeArgument = False
            self.isFieldStrideArgument = False
            self.isFieldArgument = False
            self.fieldName = ""
            self.coordinate = None

            if name.startswith(FIELD_PTR_PREFIX):
                self.isFieldPtrArgument = True
                self.isFieldArgument = True
                self.fieldName = name[len(FIELD_PTR_PREFIX):]
            elif name.startswith(FIELD_SHAPE_PREFIX):
                self.isFieldShapeArgument = True
                self.isFieldArgument = True
                self.fieldName = name[len(FIELD_SHAPE_PREFIX):]
            elif name.startswith(FIELD_STRIDE_PREFIX):
                self.isFieldStrideArgument = True
                self.isFieldArgument = True
                self.fieldName = name[len(FIELD_STRIDE_PREFIX):]
                
    def __init__(self, body):
        super(KernelFunction, self).__init__()
        self._body = body
        self._parameters = None
        self._functionName = "mykernel"
        self._body.parent = self


    @property
    def symbolsDefined(self):
        return set()

    @property
    def symbolsRead(self):
        return set()

    @property
    def parameters(self):
        self.__updateArguments()
        return self._parameters

    @property
    def args(self):
        return [self._body]

    @property
    def functionName(self):
        return self._functionName

    def __updateArguments(self):
        undefinedSymbols = self._body.symbolsRead - self._body.symbolsDefined
        self._parameters = [KernelFunction.Argument(s.name, s.dtype) for s in undefinedSymbols]
        self._parameters.sort(key=lambda l: (l.fieldName, l.isFieldPtrArgument, l.isFieldShapeArgument, l.isFieldStrideArgument, l.name),
                              reverse=True)

    def generateC(self):
        self.__updateArguments()
        functionArguments = [MyPOD(s.dtype, s.name) for s in self._parameters]
        funcDeclaration = c.FunctionDeclaration(MyPOD("void", self._functionName, ), functionArguments)
        return c.FunctionBody(funcDeclaration, self._body.generateC())


class Block(Node):
    def __init__(self, listOfNodes):
        self._nodes = listOfNodes
        for n in self._nodes:
            n.parent = self

    @property
    def args(self):
        return self._nodes

    def insertFront(self, node):
        node.parent = self
        self._nodes.insert(0, node)

    def append(self, node):
        node.parent = self
        self._nodes.append(node)

    def generateC(self):
        return c.Block([e.generateC() for e in self.args])

    def takeChildNodes(self):
        tmp = self._nodes
        self._nodes = []
        return tmp

    @property
    def symbolsDefined(self):
        result = set()
        for a in self.args:
            result.update(a.symbolsDefined)
        return result

    @property
    def symbolsRead(self):
        result = set()
        for a in self.args:
            result.update(a.symbolsRead)
        return result


class LoopOverCoordinate(Node):

    def __init__(self, body, coordinateToLoopOver, shape, increment=1, ghostLayers=1):
        self._body = body
        self._coordinateToLoopOver = coordinateToLoopOver
        self._shape = shape
        self._increment = increment
        self._ghostLayers = ghostLayers
        self._body.parent = self

    @property
    def args(self):
        result = [self._body]
        for s in self._shape:
            if isinstance(s, Node) or isinstance(s, sp.Basic):
                result.append(s)
        return result

    @property
    def loopCounterName(self):
        return "%s_%s" % (COORDINATE_LOOP_COUNTER_NAME, self._coordinateToLoopOver)

    @property
    def coordinateToLoopOver(self):
        return self._coordinateToLoopOver

    @property
    def symbolsDefined(self):
        result = self._body.symbolsDefined
        result.add(TypedSymbol(self.loopCounterName, "int"))
        return result

    @property
    def symbolsRead(self):
        result = self._body.symbolsRead
        for s in self._shape:
            if isinstance(s, sp.Basic):
                result.update(s.atoms(sp.Symbol))

        return result

    def generateC(self):
        coord = self._coordinateToLoopOver
        end = self._shape[coord] - self._ghostLayers

        counterVar = self.loopCounterName
        return c.For("int %s = %d" % (counterVar, self._ghostLayers),
                     "%s < %s" % (counterVar, codePrinter.doprint(end)),
                     "++%s" % (counterVar,),
                     self._body.generateC())


class SympyAssignment(Node):

    def __init__(self, lhsSymbol, rhsTerm, const=True):
        self._lhsSymbol = lhsSymbol
        self._rhsTerm = rhsTerm
        self._isDeclaration = True
        if isinstance(self, Field.Access):
            self._isDeclaration = False
        self._const = const

    @property
    def rhs(self):
        return self._rhsTerm

    @rhs.setter
    def rhs(self, newValue):
        self._rhsTerm = newValue

    @property
    def lhs(self):
        return self._lhsSymbol

    @lhs.setter
    def lhs(self, newValue):
        self._lhsSymbol = newValue

    @property
    def args(self):
        return [self._lhsSymbol, self._rhsTerm]

    @property
    def symbolsDefined(self):
        if not self._isDeclaration:
            return set()
        return set([self._lhsSymbol])

    @property
    def symbolsRead(self):
        return self._rhsTerm.atoms(sp.Symbol)

    def generateC(self):
        dtype = ""
        if hasattr(self._lhsSymbol, 'dtype') and self._isDeclaration:
            if self._const:
                dtype = "const " + self._lhsSymbol.dtype + " "
            else:
                dtype = self._lhsSymbol.dtype + " "

        return c.Assign(dtype + codePrinter.doprint(self._lhsSymbol),
                        codePrinter.doprint(self._rhsTerm))


class Field:
    """
    With fields one can formulate stencil-like update rules on structured grids.
    This Field class knows about the dimension, memory layout (strides) and optionally about the size of an array.

    To create a field use one of the static create* members. There are two options:
        1. create a kernel with fixed loop sizes i.e. the shape of the array is already known. This is usually the
           case if just-in-time compilation directly from Python is done. (see Field.createFromNumpyArray)
        2. create a more general kernel that works for variable array sizes. This can be used to create kernels
           beforehand for a library. (see Field.createGeneric)

    Dimensions:
        A field has spatial and index dimensions, where the spatial dimensions come first.
        The interpretation is that the field has multiple cells in (usually) two or three dimensional space which are
        looped over. Additionally  N values are stored per cell. In this case spatialDimensions is two or three,
        and indexDimensions equals N. If you want to store a matrix on each point in a two dimensional grid, there
        are four dimensions, two spatial and two index dimensions. len(arr.shape) == spatialDims + indexDims

    Indexing:
        When accessing (indexing) a field the result is a FieldAccess which is derived from sympy Symbol.
        First specify the spatial offsets in [], then in case indexDimension>0 the indices in ()
        e.g. f[-1,0,0](7)

    Example without index dimensions:
        >>> a = np.zeros([10, 10])
        >>> f = Field.createFromNumpyArray("f", a, indexDimensions=0)
        >>> jacobi = ( f[-1,0] + f[1,0] + f[0,-1] + f[0,1] ) / 4
    Example with index dimensions: LBM D2Q9 stream pull
        >>> stencil = np.array([[0,0], [0,1], [0,-1], [1,0], [-1,0], [1,1], [-1,1], [1,-1], [-1,-1]])
        >>> src = Field.createGeneric("src", spatialDimensions=2, indexDimensions=1)
        >>> dst = Field.createGeneric("dst", spatialDimensions=2, indexDimensions=1)
        >>> for i, offset in enumerate(stencil):
        >>>     print( sp.Eq(dst[0,0](i), src[-offset](i)) )
    """
    @staticmethod
    def createFromNumpyArray(fieldName, npArray, indexDimensions=0):
        """
        Creates a field based on the layout, data type, and shape of a given numpy array.
        Kernels created for these kind of fields can only be called with arrays of the same layout, shape and type.
        :param fieldName: symbolic name for the field
        :param npArray: numpy array
        :param indexDimensions: see documentation of Field
        """
        spatialDimensions = len(npArray.shape) - indexDimensions
        if spatialDimensions < 1:
            raise ValueError("Too many index dimensions. At least one spatial dimension required")

        layout = tuple(getLayoutFromNumpyField(npArray)[:spatialDimensions])
        strides = tuple([s // np.dtype(npArray.dtype).itemsize for s in npArray.strides])
        shape = tuple([int(s) for s in npArray.shape])

        return Field(fieldName, npArray.dtype, layout, shape, strides)

    @staticmethod
    def createGeneric(fieldName, spatialDimensions, dtype=np.float64, indexDimensions=0, layout=None):
        """
        Creates a generic field where the field size is not fixed i.e. can be called with arrays of different sizes
        :param fieldName: symbolic name for the field
        :param dtype: numpy data type of the array the kernel is called with later
        :param spatialDimensions: see documentation of Field
        :param indexDimensions: see documentation of Field
        :param layout: tuple specifying the loop ordering of the spatial dimensions e.g. (2, 1, 0 ) means that
                       the outer loop loops over dimension 2, the second outer over dimension 1, and the inner loop
                       over dimension 0
        """
        if not layout:
            layout = tuple(reversed(range(spatialDimensions)))
        if len(layout) != spatialDimensions:
            raise ValueError("Layout")
        shapeSymbol = IndexedBase(TypedSymbol(FIELD_SHAPE_PREFIX + fieldName, SHAPE_DTYPE), shape=(1,))
        strideSymbol = IndexedBase(TypedSymbol(FIELD_STRIDE_PREFIX + fieldName, STRIDE_DTYPE), shape=(1,))
        totalDimensions = spatialDimensions + indexDimensions
        shape = tuple([shapeSymbol[i] for i in range(totalDimensions)])
        strides = tuple([strideSymbol[i] for i in range(totalDimensions)])
        return Field(fieldName, dtype, layout, shape, strides)

    def __init__(self, fieldName, dtype, layout, shape, strides):
        """Do not use directly. Use static create* methods"""
        self._fieldName = fieldName
        self._dtype = numpyDataTypeToString(dtype)
        self._layout = layout
        self._shape = shape
        self._strides = strides

    @property
    def spatialDimensions(self):
        return len(self._layout)

    @property
    def indexDimensions(self):
        return len(self._shape) - len(self._layout)

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return self._fieldName

    @property
    def spatialShape(self):
        return self._shape[:self.spatialDimensions]

    @property
    def indexShape(self):
        return self._shape[self._spatialDimensions:]

    @property
    def spatialStrides(self):
        return self._strides[:self.spatialDimensions]

    @property
    def indexStrides(self):
        return self._strides[self.spatialDimensions:]

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, offset):
        if type(offset) is np.ndarray:
            offset = tuple(offset)
        if type(offset) is str:
            offset = tuple(directionStringToOffset(offset, self.spatialDimensions))
        if type(offset) is not tuple:
            offset = (offset,)
        if len(offset) != self.spatialDimensions:
            raise ValueError("Wrong number of spatial indices: "
                             "Got %d, expected %d" % (len(offset), self.spatialDimensions))
        return Field.Access(self, offset)

    def __hash__(self):
        return hash((self._layout, self._shape, self._strides, self._dtype, self._fieldName))

    def __eq__(self, other):
        return (self._shape, self._strides, self.name, self.dtype) == (other._shape, other._strides, other.name, other.dtype)

    class Access(sp.Symbol):
        def __new__(cls, name, *args, **kwds):
            obj = Field.Access.__xnew_cached_(cls, name, *args, **kwds)
            return obj

        def __new_stage2__(cls, field, offsets=(0, 0, 0), idx=None):
            fieldName = field.name
            offsetName = offsetToDirectionString(offsets)
            offsetVector = np.array(offsets)
            if not idx:
                idx = tuple([0] * field.indexDimensions)

            if field.indexDimensions == 0:
                obj = super(Field.Access, cls).__xnew__(cls, fieldName + "_" + offsetName)
            elif field.indexDimensions == 1:
                obj = super(Field.Access, cls).__xnew__(cls, fieldName + "_" + offsetName + "^" + str(idx[0]))
            else:
                idxStr = ",".join([str(e) for e in idx])
                obj = super(Field.Access, cls).__xnew__(cls, fieldName + "_" + offsetName + "^" + idxStr)

            obj._field = field
            obj._offsets = [int(i) for i in offsetVector]
            obj._offsetName = offsetName
            obj._index = idx

            return obj

        __xnew__ = staticmethod(__new_stage2__)
        __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

        def __call__(self, *idx):
            if self._index != tuple([0]*self.field.indexDimensions):
                print(self._index, tuple([0]*self.field.indexDimensions))
                raise ValueError("Indexing an already indexed Field.Access")

            idx = tuple(idx)
            if len(idx) != self.field.indexDimensions:
                raise ValueError("Wrong number of indices: "
                                 "Got %d, expected %d" % (len(idx), self.field.indexDimensions))
            return Field.Access(self.field, self._offsets, idx)

        @property
        def field(self):
            return self._field

        @property
        def offsets(self):
            return self._offsets

        @property
        def requiredGhostLayers(self):
            return int(np.max(np.abs(self._offsets)))

        @property
        def nrOfCoordinates(self):
            return len(self._offsets)

        @property
        def index(self):
            return self._index

        def _hashable_content(self):
            superClassContents = list(super(Field.Access, self)._hashable_content())
            t = tuple([*superClassContents, hash(self._field), self._index] + self._offsets)
            return t


class TypedSymbol(sp.Symbol):

    def __new__(cls, name, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype):
        obj = super(TypedSymbol, cls).__xnew__(cls, name)
        obj._dtype = dtype

        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self._dtype


# --------------------------------------- Factory Functions ------------------------------------------------------------


def makeLoopOverDomain(body):
    """
    :param body: list of nodes
    :return: LoopOverCoordinate instance with nested loops, ordered according to field layouts
    """

    # find correct ordering by inspecting participating FieldAccesses
    fieldAccesses = body.atoms(Field.Access)
    fields = set([e.field for e in fieldAccesses])
    layouts = set([field.layout for field in fields])
    if len(layouts) > 1:
        warnings.warn("makeLoopOverDomain: Due to different layout of the fields no optimal loop ordering exists")
    layout = list(layouts)[0]

    # find number of required ghost layers
    requiredGhostLayers = max([fa.requiredGhostLayers for fa in fieldAccesses])

    shapes = set([f.spatialShape for f in fields])

    if len(shapes) > 1:
        nrOfFixedSizedFields = 0
        for shape in shapes:
            if not isinstance(shape[0], sp.Basic):
                nrOfFixedSizedFields += 1
        assert nrOfFixedSizedFields <= 1, "Differently sized field accesses in loop body: " + str(shapes)
    shape = list(shapes)[0]

    currentBody = body
    for loopCoordinate in reversed(layout):
        currentBody = Block([LoopOverCoordinate(currentBody, loopCoordinate, shape, 1, requiredGhostLayers)])

    return KernelFunction(currentBody)


# --------------------------------------- Transformations --------------------------------------------------------------


def resolveFieldAccesses(ast):
    """Substitutes FieldAccess nodes by array indexing"""

    def visitSympyExpr(expr, enclosingBlock, enclosingLoop):
        if isinstance(expr, Field.Access):
            fieldAccess = expr
            field = fieldAccess.field
            dtype = "%s *" % field.dtype
            fieldPtrType = "%s *" % (field.dtype,)
            fieldPtr = TypedSymbol("%s%s" % (FIELD_PTR_PREFIX, field.name), fieldPtrType)
            idxStr = "_".join([str(i) for i in fieldAccess.index])
            basePtr = TypedSymbol("%s%s_%s" % (BASE_PTR_PREFIX, field.name, idxStr), dtype)
            baseArr = IndexedBase(basePtr, shape=(1,))

            offset = 0
            fastestLoopCoord = enclosingLoop.coordinateToLoopOver
            for i in range(field.spatialDimensions):
                if i == fastestLoopCoord:
                    continue
                offset += field.spatialStrides[i] * TypedSymbol("%s_%d" % (COORDINATE_LOOP_COUNTER_NAME, i), "int")
            for i in range(field.indexDimensions):
                offset += field.indexStrides[i] * fieldAccess.index[i]

            if basePtr not in enclosingBlock.symbolsDefined:
                enclosingBlock.insertFront(SympyAssignment(basePtr, fieldPtr + offset, const=False))

            neighborOffset = sum([field.spatialStrides[c] * fieldAccess.offsets[c]
                                  for c in range(len(fieldAccess.offsets))])
            innerCounterVar = TypedSymbol("%s_%d" % (COORDINATE_LOOP_COUNTER_NAME, fastestLoopCoord), "int")
            return baseArr[innerCounterVar*field.spatialStrides[fastestLoopCoord] + neighborOffset]
        else:
            newArgs = [visitSympyExpr(e, enclosingBlock, enclosingLoop) for e in expr.args]
            kwargs = {'evaluate': False} if type(expr) == sp.Add or type(expr) == sp.Mul else {}
            return expr.func(*newArgs, **kwargs) if newArgs else expr

    def visitNode(subAst, enclosingBlock, enclosingLoop):
        if isinstance(subAst, SympyAssignment):
            subAst.lhs = visitSympyExpr(subAst.lhs, enclosingBlock, enclosingLoop)
            subAst.rhs = visitSympyExpr(subAst.rhs, enclosingBlock, enclosingLoop)
        else:
            for i, a in enumerate(subAst.args):
                visitNode(a,
                          subAst if isinstance(subAst, Block) else enclosingBlock,
                          subAst if isinstance(subAst, LoopOverCoordinate) else enclosingLoop)

    return visitNode(ast, None, None)


def moveConstantsBeforeLoop(ast):

    def findBlockToMoveTo(node):
        """Traverses parents of node as long as the symbols are independent and returns a (parent) block
        the assignment can be safely moved to
        :param node: SympyAssignment inside a Block"""
        assert isinstance(node, SympyAssignment)
        assert isinstance(node.parent, Block)

        lastBlock = node.parent
        element = node.parent
        while element:
            if isinstance(element, Block):
                lastBlock = element
            if node.symbolsRead.intersection(element.symbolsDefined):
                break
            element = element.parent
        return lastBlock

    for block in ast.atoms(Block):
        children = block.takeChildNodes()
        for child in children:
            if not isinstance(child, SympyAssignment):
                block.append(child)
            else:
                target = findBlockToMoveTo(child)
                if target == block:     # movement not possible
                    target.append(child)
                else:
                    target.insertFront(child)


# ------------------------------------- Main ---------------------------------------------------------------------------


def createKernel(listOfEquations, functionName="kernel", typeForSymbol=defaultdict(lambda: "double")):

    fieldsWritten = set()
    fieldsRead = set()
    explicitReadAssignments = []
    symbolsWithReadAssignment = set()

    def replaceCharactersForC(s):
        return s.replace("^", "_")

    def processRhs(term):
        """Replaces Symbols by:
            - new variable and defines the new variable in explicitReadAssignments (for field accesses)
              type is used from the field, additionally adds the field to fieldsRead
            - TypedSymbol if symbol is not a field access
        """
        if isinstance(term, Field.Access):
            fieldsRead.add(term.field)
            substitute = TypedSymbol(replaceCharactersForC(term.name), term.field.dtype)
            if substitute not in symbolsWithReadAssignment:
                explicitReadAssignments.append(SympyAssignment(substitute, term))
                symbolsWithReadAssignment.add(substitute)
            return substitute
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            newArgs = [processRhs(arg) for arg in term.args]
            return term.func(*newArgs) if newArgs else term

    def processLhs(term):
        """Replaces symbol by TypedSymbol and adds field to fieldsWriten"""
        if isinstance(term, Field.Access):
            fieldsWritten.add(term.field)
            return term
        elif isinstance(term, sp.Symbol):
            return TypedSymbol(term.name, typeForSymbol[term.name])
        else:
            assert False, "Expected a symbol as left-hand-side"

    assignments = []
    for eq in listOfEquations:
        newLhs = processLhs(eq.lhs)
        newRhs = processRhs(eq.rhs)
        assignments.append(SympyAssignment(newLhs, newRhs))
    assignments = explicitReadAssignments + assignments

    body = Block(assignments)
    code = makeLoopOverDomain(body)
    resolveFieldAccesses(code)
    moveConstantsBeforeLoop(code)
    return code
