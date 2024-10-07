set(VECTORDB_BENCH_SOURCES vectordb_bench.cpp)
add_executable(vectordb-bench ${VECTORDB_BENCH_SOURCES})

target_link_libraries(vectordb-bench bustub)
set_target_properties(vectordb-bench PROPERTIES OUTPUT_NAME bustub-vectordb-bench)
