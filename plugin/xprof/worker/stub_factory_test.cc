#include "plugin/xprof/worker/stub_factory.h"

#include <thread>  // NOLINT
#include <vector>

#include "<gtest/gtest.h>"

namespace xprof {
namespace profiler {
namespace {

class StubFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override { internal::ResetStubsForTesting(); }
};

TEST_F(StubFactoryTest, NoStubs) { EXPECT_EQ(GetNextStub(), nullptr); }

TEST_F(StubFactoryTest, InitializeAndGetNextStub) {
  InitializeStubs("localhost:1234,localhost:5678");
  auto stub1 = GetNextStub();
  auto stub2 = GetNextStub();
  auto stub3 = GetNextStub();
  EXPECT_NE(stub1, nullptr);
  EXPECT_NE(stub2, nullptr);
  EXPECT_NE(stub3, nullptr);
  EXPECT_EQ(stub1, stub3);
}

TEST_F(StubFactoryTest, ConcurrentGetNextStub) {
  InitializeStubs("localhost:1000,localhost:2000,localhost:3000");
  constexpr int kNumThreads = 10;
  constexpr int kNumCallsPerThread = 100;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < kNumCallsPerThread; ++j) {
        EXPECT_NE(GetNextStub(), nullptr);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(StubFactoryTest, ConcurrentInitialize) {
  constexpr int kNumThreads = 10;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(
        [&]() { InitializeStubs("localhost:4000,localhost:5000"); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  auto stub1 = GetNextStub();
  auto stub2 = GetNextStub();
  auto stub3 = GetNextStub();
  EXPECT_NE(stub1, nullptr);
  EXPECT_NE(stub2, nullptr);
  EXPECT_NE(stub3, nullptr);
  EXPECT_EQ(stub1, stub3);
}

TEST_F(StubFactoryTest, InitializeWithEmptyString) {
  InitializeStubs("");
  EXPECT_EQ(GetNextStub(), nullptr);
}

TEST_F(StubFactoryTest, InitializeWithMalformedString) {
  InitializeStubs("localhost:1111,,localhost:2222,");

  auto stub1 = GetNextStub();
  auto stub2 = GetNextStub();

  EXPECT_NE(stub1, nullptr);
  EXPECT_NE(stub2, nullptr);
  EXPECT_NE(stub1, stub2);
}

TEST_F(StubFactoryTest, ReinitializationIsIgnored) {
  InitializeStubs("localhost:1111");
  EXPECT_NE(GetNextStub(), nullptr);

  InitializeStubs("");

  EXPECT_NE(GetNextStub(), nullptr);
}

TEST_F(StubFactoryTest, ResetClearsStubs) {
  InitializeStubs("localhost:1234");
  EXPECT_NE(GetNextStub(), nullptr);

  internal::ResetStubsForTesting();

  EXPECT_EQ(GetNextStub(), nullptr);

  InitializeStubs("localhost:5678");
  EXPECT_NE(GetNextStub(), nullptr);
}

}  // namespace
}  // namespace profiler
}  // namespace xprof
