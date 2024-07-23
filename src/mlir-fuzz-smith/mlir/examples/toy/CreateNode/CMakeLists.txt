# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
)

# set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
# mlir_tablegen(ToyCombine.inc -gen-rewriters)
# add_public_tablegen_target(CreateOperationOPsIncGen)

add_toy_chapter(CreateNode
  main.cpp
  ArithNodeGenerator.cpp
  AffineNodeGenerator.cpp
  BufferizationNodeGenerator.cpp
  IndexNodeGenerator.cpp
  LinalgNodeGenerator.cpp
  MathNodeGenerator.cpp
  MemrefNodeGenerator.cpp
  SCFNodeGenerator.cpp
  SpirvNodeGenerator.cpp
  TensorNodeGenerator.cpp
  TosaNodeGenerator.cpp
  VectorNodeGenerator.cpp
  ValuePool.cpp
  TypeBase.cpp
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(CreateNode
  PRIVATE
    ${dialect_libs}
    ${extension_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms)
