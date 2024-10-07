/**
    Base code is from Faiss and modified to adapt to bustub.
 */

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "common/bustub_instance.h"
#include "storage/index/hnsw_index.h"

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

auto FvecsRead(const char *fname, size_t *d_out, size_t *n_out) -> float * {
  FILE *f = fopen(fname, "r");
  if (f == nullptr) {
    fprintf(stderr, "could not open %s\n", fname);
    perror("");
    abort();
  }
  int d;
  fread(&d, 1, sizeof(int), f);
  assert((d > 0 && d < 1000000) || !"unreasonable dimension");
  fseek(f, 0, SEEK_SET);
  struct stat st;
  fstat(fileno(f), &st);
  size_t sz = st.st_size;
  assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
  size_t n = sz / ((d + 1) * 4);

  *d_out = d;
  *n_out = n;
  auto *x = new float[n * (d + 1)];
  size_t nr = fread(x, sizeof(float), n * (d + 1), f);
  assert(nr == n * (d + 1) || !"could not read whole file");

  // shift array to remove row headers
  for (size_t i = 0; i < n; i++) {
    memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
  }

  fclose(f);
  return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
auto IvecsRead(const char *fname, size_t *d_out, size_t *n_out) -> int * {
  return reinterpret_cast<int *>(FvecsRead(fname, d_out, n_out));
}

auto Elapsed() -> double {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

auto Vector2String(const float *p, int n) -> std::string {
  std::stringstream ss;
  ss << '[';

  for (int i = 0; i < n; ++i) {
    ss << std::fixed << std::setprecision(6) << p[i];

    if (i < n - 1) {
      ss << ", ";
    }
  }

  ss << ']';
  return ss.str();
}

void InsertIndexVectorData(bustub::BustubInstance &bustub, bustub::ResultWriter &writer, double t0) {
  // create index(first insert data, then create index maybe better)
  printf("[%.3f s] Creating vector index...\n", Elapsed() - t0);
  // HNSW index, use the default parameter
  std::string create_index =
      "CREATE INDEX t1v1hnsw ON t1 USING hnsw (v1 vector_l2_ops) WITH (m = 16, ef_construction = 64, ef_search = 100);";

  bool status = bustub.ExecuteSql(create_index, writer);

  assert(status || !"Failed to create vector index.");

  // insert data
  printf("[%.3f s] Loading database\n", Elapsed() - t0);

  size_t nb;
  size_t d2;
  float *xb = FvecsRead("sift1M/sift_base.fvecs", &d2, &nb);
  assert(d2 == 128 || !"vector dimension should be 128.");

  printf("[%.3f s] Loading database, size %ld*%ld\n", Elapsed() - t0, nb, d2);

  std::string insert_sql = "INSERT INTO t1 VALUES (ARRAY ";
  for (size_t i = 0; i < nb; ++i) {
    if (i % 1000 == 0) {
      printf("[%.3f s] Loading database, #%ld  #%ld\n", Elapsed() - t0, i, nb);
    }
    // read vector and insert to table
    const std::string vector = Vector2String(xb + i * d2, d2);
    std::string sql = insert_sql + vector + " , " + std::to_string(i) + ");";
    bool status = bustub.ExecuteSql(sql, writer);
    if (!status) {
      std::cerr << "Insert data failed: index = " << i << std::endl;
    }
  }
  delete[] xb;
}

class Metric {
 public:
  Metric() = default;

  void AddQueryResult(const std::vector<int> &res, const int *gt) {
    ++n_;
    // just care about the nearest vector, modify according to your requirements
    int target = gt[0];
    for (size_t i = 0; i < res.size(); ++i) {
      if (res[i] == target) {
        if (i < 1) {
          ++n_1_;
        }
        if (i < 10) {
          ++n_10_;
        }
        if (i < 100) {
          ++n_100_;
        }
        break;
      }
    }
  }

  void Show() const {
    printf("R@1 = %.4f\n", n_1_ / static_cast<float>(n_));
    printf("R@10 = %.4f\n", n_10_ / static_cast<float>(n_));
    printf("R@100 = %.4f\n", n_100_ / static_cast<float>(n_));
  }

 private:
  size_t n_1_{0};
  size_t n_10_{0};
  size_t n_100_{0};
  size_t n_{0};
};

void DoANNQuery(bustub::BustubInstance &bustub, bustub::StringVectorWriter &writer, double t0, const float *p,
                size_t nq, const int *gt) {
  Metric metric;
  std::string select_sql = "SELECT v2, v1 FROM t1 ORDER BY ARRAY ";
  for (size_t i = 0; i < nq; ++i) {
    if (i % 1000 == 0) {
      printf("[%.3f s] Doing query, #%ld  #%ld\n", Elapsed() - t0, i, nq);
    }
    std::string sql = select_sql + Vector2String(p + 128 * i, 128) + " <-> v1 LIMIT 100;";
    bustub.ExecuteSql(sql, writer);
    auto &res = writer.values_;
    std::vector<int> id_lst;
    id_lst.reserve(100);
    for (const auto &lines : res) {
      id_lst.push_back(std::stoi(lines[0]));
    }
    // calculate recall
    metric.AddQueryResult(id_lst, gt + 100 * i);
  }
  printf("[%.3f s] Compute recalls\n", Elapsed() - t0);
  metric.Show();
}

auto main() -> int {
  double t0 = Elapsed();

  bustub::BustubInstance bustub(128 * 1024);
  auto writer = bustub::NoopWriter();

  // create table
  const std::string create_table_sql = "CREATE TABLE t1(v1 VECTOR(128), v2 integer);";
  bool status = bustub.ExecuteSql(create_table_sql, writer);
  assert(status || !"Failed to create table.");

  // insert data
  InsertIndexVectorData(bustub, writer, t0);

  // read query vectors
  size_t nq;
  float *xq;

  {
    printf("[%.3f s] Loading queries\n", Elapsed() - t0);

    size_t d2;
    xq = FvecsRead("sift1M/sift_query.fvecs", &d2, &nq);
    assert(d2 == 128 || !"query vector dimension should be 128");
  }

  size_t k;  // nb of results per query in the GT
  int *gt;   // nq * k matrix of ground-truth nearest-neighbors

  {
    printf("[%.3f s] Loading ground truth for %ld queries\n", Elapsed() - t0, nq);

    // load ground-truth and convert int to long
    size_t nq2;
    int *gt_int = IvecsRead("sift1M/sift_groundtruth.ivecs", &k, &nq2);
    assert(nq2 == nq || !"incorrect nb of ground truth entries");
    gt = new int[k * nq];
    for (size_t i = 0; i < k * nq; i++) {
      gt[i] = gt_int[i];
    }
    delete[] gt_int;
  }

  // do query
  bustub::StringVectorWriter string_vector_writer = bustub::StringVectorWriter();
  DoANNQuery(bustub, string_vector_writer, t0, xq, nq, gt);

  delete[] xq;
  delete[] gt;
  return 0;
}
