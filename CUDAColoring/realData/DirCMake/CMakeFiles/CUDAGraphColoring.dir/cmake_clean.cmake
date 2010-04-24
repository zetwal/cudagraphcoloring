FILE(REMOVE_RECURSE
  "./CUDAGraphColoring_generated_graphColoring.cu.o"
  "libCUDAGraphColoring.pdb"
  "libCUDAGraphColoring.so"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/CUDAGraphColoring.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
