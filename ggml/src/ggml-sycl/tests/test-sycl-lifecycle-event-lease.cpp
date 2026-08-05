#include "execution-lifecycle.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace ggml_sycl::execution;
using ggml_sycl::lifecycle::LoadTxnId;
using ggml_sycl::lifecycle::ModelId;
using ggml_sycl::lifecycle::ModelToken;
using ggml_sycl::lifecycle::SlotToken;

static void require(bool value, const char * message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static ModelToken root_token(uint64_t base) {
    return { ModelId{ base }, LoadTxnId{ base + 1000 }, SlotToken{ static_cast<uint32_t>(base % 13), base + 2000 } };
}

static void g5a() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "G5a create failed");
    require(reg.bind_backend(ctx,0)==error::OK && reg.bind_backend(ctx,1)==error::OK, "G5a bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(50); require(reg.attach_root(ctx, root, &s, &e)==error::OK, "G5a attach failed");
    GraphEpoch g{}; InvocationId i{}; const int devices[] = {0,1}; snapshot snap{};
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK && reg.begin_invocation(ctx,s,e,g,root,devices,2,&i)==error::OK, "G5a invoke failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.busy_device_count==2, "G5a busy aggregate failed");
    require(reg.seal_invocation(ctx,s,e,g,i,root)==error::OK, "G5a seal failed");
    require(reg.complete_invocation(ctx,s,e,g,i,root,0)==error::OK, "G5a dev0 complete failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.graph_state==graph_phase::COMPLETE && snap.busy_device_count==2 && snap.invocation.value!=0, "G5a first sync gap");
    require(reg.complete_invocation(ctx,s,e,g,i,root,1)==error::OK, "G5a dev1 complete failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.busy_device_count==0 && snap.invocation.value==0, "G5a final release failed");
}

static void g6() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "G6 create failed");
    require(reg.bind_backend(ctx,0)==error::OK && reg.bind_backend(ctx,1)==error::OK, "G6 bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(60); require(reg.attach_root(ctx, root, &s, &e)==error::OK, "G6 attach failed");
    GraphEpoch g{}; InvocationId i{}; const int devices[] = {0,1}; snapshot snap{};
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK && reg.begin_invocation(ctx,s,e,g,root,devices,2,&i)==error::OK, "G6 invoke failed");
    require(reg.seal_invocation(ctx,s,e,g,i,root)==error::OK, "G6 seal failed");
    require(reg.complete_invocation(ctx,s,e,g,i,root,0)==error::OK, "G6 first sync failed");
    require(reg.begin_graph(ctx,s,e,root,&g)==error::BUSY, "G6 terminal overwritten before final sync");
    require(reg.complete_invocation(ctx,s,e,g,i,root,1)==error::OK, "G6 second sync failed");
    require(reg.retire_graph(ctx,s,e,g,root)==error::OK, "G6 retire failed");
}

static void g7() {
    Registry reg; error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "G7 create failed");
    require(reg.bind_backend(a,2)==error::OK && reg.bind_backend(b,2)==error::OK, "G7 bind failed");
    SessionId sa{}, sb{}; SessionResetEpoch ea{}, eb{}; auto ra = root_token(70), rb = root_token(71);
    require(reg.attach_root(a,ra,&sa,&ea)==error::OK && reg.attach_root(b,rb,&sb,&eb)==error::OK, "G7 attach failed");
    GraphEpoch ga{}, gb{}; InvocationId ia{}, ib{}; const int d[] = {2};
    require(reg.begin_graph(a,sa,ea,ra,&ga)==error::OK && reg.begin_invocation(a,sa,ea,ga,ra,d,1,&ia)==error::OK, "G7 invoke A failed");
    require(reg.begin_graph(b,sb,eb,rb,&gb)==error::OK && reg.begin_invocation(b,sb,eb,gb,rb,d,1,&ib)==error::DEVICE_BUSY, "G7 missing same-device busy");
    require(reg.complete_invocation(a,sa,ea,ga,ia,ra,2)==error::OK, "G7 complete A failed");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,&ib)==error::OK, "G7 did not recover after release");
}

int main(int argc, char ** argv) {
    const char * which = argc > 2 && std::strcmp(argv[1], "--case") == 0 ? argv[2] : "all";
    if (std::strcmp(which, "G5a") == 0) g5a();
    else if (std::strcmp(which, "G6") == 0) g6();
    else if (std::strcmp(which, "G7") == 0) g7();
    else { g5a(); g6(); g7(); }
    return 0;
}
