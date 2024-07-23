# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_mlir_tool(mlirsmith
  toyc.cpp
  mlir/Dialect.cpp
  smith/MLIRSmith.cpp
  smith/generators/OpGeneratorProvider.cpp
  smith/generators/ArithGeneratorProvider.cpp
  smith/generators/BufferizationGeneratorProvider.cpp
  smith/generators/IndexGeneratorProvider.cpp
  smith/generators/LinalgGeneratorProvider.cpp
  smith/generators/MathGeneratorProvider.cpp
  smith/generators/MemrefGeneratorProvider.cpp
  smith/generators/FuncGeneratorProvider.cpp
  smith/generators/SCFGeneratorProvider.cpp
  smith/generators/SPIRVGeneratorProvider.cpp
  smith/generators/TensorGeneratorProvider.cpp
  smith/generators/VectorGeneratorProvider.cpp
  smith/ValueGeneration.cpp
  smith/Util.cpp
  smith/RegionGeneration.cpp
  smith/TypedValuePool.cpp
  smith/TypeValue.cpp
  smith/generators/AffineGeneratorProvider.cpp
  smith/OpPrinterPass.cpp
  smith/TmplInstantiation.cpp
  DEPENDS
  MLIRSmithOpsIncGen
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(mlirsmith
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
