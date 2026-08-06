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
    GraphEpoch g{}; InvocationId i{}; const int devices[] = {0,1}; const int participants[] = {7}; snapshot snap{};
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK && reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,1,7,&i)==error::OK, "G5a invoke failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.busy_device_count==2, "G5a busy aggregate failed");
    require(reg.seal_invocation(ctx,s,e,g,i,root)==error::OK, "G5a seal failed");
    require(reg.complete_invocation(ctx,s,e,g,i,root,7)==error::OK, "G5a participant complete failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.graph_state==graph_phase::COMPLETE && snap.busy_device_count==0 && snap.invocation.value==0, "G5a final release failed");
}

static void g6() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "G6 create failed");
    require(reg.bind_backend(ctx,0)==error::OK && reg.bind_backend(ctx,1)==error::OK, "G6 bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(60); require(reg.attach_root(ctx, root, &s, &e)==error::OK, "G6 attach failed");
    GraphEpoch g{}; InvocationId i{}; InvocationId j{}; const int devices[] = {0,1}; const int participants[] = {5,9};
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK && reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,5,&i)==error::OK, "G6 invoke failed");
    require(reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,9,&j)==error::OK && j.value == i.value, "G6 join failed");
    require(reg.seal_invocation(ctx,s,e,g,i,root)==error::OK, "G6 seal failed");
    require(reg.complete_invocation(ctx,s,e,g,i,root,9)==error::OK, "G6 first sync failed");
    require(reg.begin_graph(ctx,s,e,root,&g)==error::BUSY, "G6 terminal overwritten before final sync");
    require(reg.complete_invocation(ctx,s,e,g,i,root,5)==error::OK, "G6 second sync failed");
    require(reg.retire_graph(ctx,s,e,g,root)==error::OK, "G6 retire failed");
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK, "G6 next graph blocked after retire");
}

static void g6b() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "G6b create failed");
    require(reg.bind_backend(ctx,0)==error::OK && reg.bind_backend(ctx,1)==error::OK, "G6b bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(61); require(reg.attach_root(ctx, root, &s, &e)==error::OK, "G6b attach failed");
    GraphEpoch g{}; InvocationId i{}; InvocationId j{}; const int devices[] = {0,1}; const int participants[] = {11,17}; snapshot snap{};
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK && reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,11,&i)==error::OK, "G6b invoke A failed");
    require(reg.seal_invocation(ctx,s,e,g,i,root)==error::OK, "G6b seal failed");
    require(reg.complete_invocation(ctx,s,e,g,i,root,11)==error::OK, "G6b early complete failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.invocation.value!=0, "G6b finalized before reserved join");
    require(reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,17,&j)==error::OK && j.value==i.value, "G6b join after completion failed");
    require(reg.quarantine_invocation(ctx,s,e,g,i,root,17)==error::OK, "G6b final quarantine failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.invocation.value==0 && snap.graph_state==graph_phase::QUARANTINED, "G6b final release failed");
}

static void g7() {
    Registry reg; error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "G7 create failed");
    require(reg.bind_backend(a,2)==error::OK && reg.bind_backend(b,2)==error::OK, "G7 bind failed");
    SessionId sa{}, sb{}; SessionResetEpoch ea{}, eb{}; auto ra = root_token(70), rb = root_token(71);
    require(reg.attach_root(a,ra,&sa,&ea)==error::OK && reg.attach_root(b,rb,&sb,&eb)==error::OK, "G7 attach failed");
    GraphEpoch ga{}, gb{}; InvocationId ia{}, ib{}; const int d[] = {2}; const int p[] = {31};
    require(reg.begin_graph(a,sa,ea,ra,&ga)==error::OK && reg.begin_invocation(a,sa,ea,ga,ra,d,1,p,1,31,&ia)==error::OK, "G7 invoke A failed");
    require(reg.begin_graph(b,sb,eb,rb,&gb)==error::OK && reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,31,&ib)==error::DEVICE_BUSY, "G7 missing same-device busy");
    require(reg.complete_invocation(a,sa,ea,ga,ia,ra,31)==error::OK, "G7 complete A failed");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,31,&ib)==error::OK, "G7 did not recover after release");
}

static void h8() {
    Registry reg; error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "H8 create failed");
    require(reg.bind_backend(a,2)==error::OK && reg.bind_backend(b,2)==error::OK, "H8 bind failed");
    SessionId sa{}, sb{}; SessionResetEpoch ea{}, eb{}; auto ra = root_token(80), rb = root_token(81);
    require(reg.attach_root(a,ra,&sa,&ea)==error::OK && reg.attach_root(b,rb,&sb,&eb)==error::OK, "H8 attach failed");
    GraphEpoch ga{}, gb{}; InvocationId ia{}, ib{}; const int d[] = {2}; const int p[] = {37}; snapshot snap{};
    require(reg.begin_graph(a,sa,ea,ra,&ga)==error::OK && reg.begin_invocation(a,sa,ea,ga,ra,d,1,p,1,37,&ia)==error::OK, "H8 invoke A failed");
    require(reg.begin_graph(b,sb,eb,rb,&gb)==error::OK, "H8 graph B failed");
    require(reg.submit_invocation(a,sa,ea,ga,ia,ra,37)==error::OK, "H8 submit failed");
    require(reg.extract(a,&snap)==error::OK && snap.graph_state==graph_phase::COMPLETE && snap.busy_device_count==1 && snap.invocation.value==ia.value,
            "H8 submit did not retain execution token");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,37,&ib)==error::DEVICE_BUSY, "H8 submitted event lost device hold");
    require(reg.release_invocation(a,sa,ea,ga,ia,ra)==error::OK, "H8 release failed");
    require(reg.retire_graph(a,sa,ea,ga,ra)==error::OK, "H8 retire failed");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,37,&ib)==error::OK, "H8 did not recover after event release");
}

static void h12() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "H12 create failed");
    require(reg.bind_backend(ctx,0)==error::OK && reg.bind_backend(ctx,1)==error::OK, "H12 bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(120); require(reg.attach_root(ctx, root, &s, &e)==error::OK, "H12 attach failed");
    GraphEpoch g{}; InvocationId i{}; const int devices[] = {0,1}; const int participants[] = {41,43}; snapshot snap{}; ResetTicket rt{}; DrainTicket dt{};
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK && reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,41,&i)==error::OK, "H12 invoke A failed");
    InvocationId j{}; require(reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,43,&j)==error::OK && j.value == i.value, "H12 invoke B failed");
    require(reg.submit_invocation(ctx,s,e,g,i,root,43)==error::OK, "H12 first submit failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.graph_state==graph_phase::OPEN && snap.token_root==root, "H12 first submit lost aggregate root");
    require(reg.submit_invocation(ctx,s,e,g,i,root,41)==error::OK, "H12 terminal submit failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.graph_state==graph_phase::COMPLETE && snap.token_root_state==token_root_phase::COMPLETE &&
                snap.busy_device_count==2 && snap.invocation.value==i.value,
            "H12 terminal submit did not retain aggregate graph root");
    require(reg.begin_graph(ctx,s,e,root,&g)==error::BUSY, "H12 terminal graph reopened before release");
    require(reg.begin_reset(ctx,s,e,&rt)==error::BUSY, "H12 reset ignored submitted event token");
    require(reg.begin_drain(ctx,&dt)==error::BUSY, "H12 drain ignored submitted event token");
    require(reg.release_invocation(ctx,s,e,g,i,root)==error::OK, "H12 release failed");
    require(reg.retire_graph(ctx,s,e,g,root)==error::OK, "H12 retire failed");
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK, "H12 next graph blocked after release");
}

static void h12b() {
    Registry reg; error err = error::OK; const auto ctx = reg.create_context(err); require(err == error::OK, "H12b create failed");
    require(reg.bind_backend(ctx,0)==error::OK && reg.bind_backend(ctx,1)==error::OK, "H12b bind failed");
    SessionId s{}; SessionResetEpoch e{}; auto root = root_token(121); require(reg.attach_root(ctx, root, &s, &e)==error::OK, "H12b attach failed");
    GraphEpoch g{}; InvocationId i{}; const int devices[] = {0,1}; const int participants[] = {45,47}; snapshot snap{}; ResetTicket rt{}; DrainTicket dt{};
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK && reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,45,&i)==error::OK, "H12b invoke A failed");
    InvocationId j{}; require(reg.begin_invocation(ctx,s,e,g,root,devices,2,participants,2,47,&j)==error::OK && j.value == i.value, "H12b invoke B failed");
    require(reg.submit_invocation(ctx,s,e,g,i,root,47)==error::OK, "H12b first submit failed");
    require(reg.submit_quarantined_invocation(ctx,s,e,g,i,root,45)==error::OK, "H12b terminal quarantine failed");
    require(reg.extract(ctx,&snap)==error::OK && snap.graph_state==graph_phase::QUARANTINED &&
                snap.token_root_state==token_root_phase::QUARANTINED && snap.busy_device_count==2 &&
                snap.invocation.value==i.value && snap.token_root==root,
            "H12b terminal quarantine did not retain aggregate graph root");
    require(reg.begin_graph(ctx,s,e,root,&g)==error::BUSY, "H12b quarantined graph reopened before release");
    require(reg.begin_reset(ctx,s,e,&rt)==error::BUSY, "H12b reset ignored quarantined event token");
    require(reg.begin_drain(ctx,&dt)==error::BUSY, "H12b drain ignored quarantined event token");
    require(reg.release_invocation(ctx,s,e,g,i,root)==error::OK, "H12b release failed");
    require(reg.retire_graph(ctx,s,e,g,root)==error::OK, "H12b retire failed");
    require(reg.begin_graph(ctx,s,e,root,&g)==error::OK, "H12b next graph blocked after release");
}

static void h13b() {
    Registry reg; error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "H13b create failed");
    require(reg.bind_backend(a,2)==error::OK && reg.bind_backend(b,2)==error::OK, "H13b bind failed");
    SessionId sa{}, sb{}; SessionResetEpoch ea{}, eb{}; auto ra = root_token(130), rb = root_token(131);
    require(reg.attach_root(a,ra,&sa,&ea)==error::OK && reg.attach_root(b,rb,&sb,&eb)==error::OK, "H13b attach failed");
    GraphEpoch ga{}, gb{}; InvocationId ia{}, ib{}; const int d[] = {2}; const int p[] = {49}; snapshot snap{};
    require(reg.begin_graph(a,sa,ea,ra,&ga)==error::OK && reg.begin_invocation(a,sa,ea,ga,ra,d,1,p,1,49,&ia)==error::OK, "H13b invoke A failed");
    require(reg.begin_graph(b,sb,eb,rb,&gb)==error::OK, "H13b graph B failed");
    require(reg.submit_invocation(a,sa,ea,ga,ia,ra,49)==error::OK, "H13b submit failed");
    require(reg.release_invocation(a,sa,ea,ga,ia,rb)==error::MISMATCH, "H13b release accepted wrong root");
    require(reg.extract(a,&snap)==error::OK && snap.busy_device_count==1 && snap.invocation.value==ia.value,
            "H13b wrong-root release damaged retained execution ownership");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,49,&ib)==error::DEVICE_BUSY, "H13b wrong-root release freed competing context");
    require(reg.release_invocation(a,sa,ea,ga,ia,ra)==error::OK, "H13b exact owner release failed");
    require(reg.retire_graph(a,sa,ea,ga,ra)==error::OK, "H13b retire failed");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,49,&ib)==error::OK, "H13b competing context did not recover after exact release");
}

static void m7() {
    Registry reg(test_mutation::M7_SUBMIT_RELEASES_DEVICES_EARLY); error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "M7 create failed");
    require(reg.bind_backend(a,2)==error::OK && reg.bind_backend(b,2)==error::OK, "M7 bind failed");
    SessionId sa{}, sb{}; SessionResetEpoch ea{}, eb{}; auto ra = root_token(90), rb = root_token(91);
    require(reg.attach_root(a,ra,&sa,&ea)==error::OK && reg.attach_root(b,rb,&sb,&eb)==error::OK, "M7 attach failed");
    GraphEpoch ga{}, gb{}; InvocationId ia{}, ib{}; const int d[] = {2}; const int p[] = {53}; snapshot snap{};
    require(reg.begin_graph(a,sa,ea,ra,&ga)==error::OK && reg.begin_invocation(a,sa,ea,ga,ra,d,1,p,1,53,&ia)==error::OK, "M7 invoke A failed");
    require(reg.begin_graph(b,sb,eb,rb,&gb)==error::OK, "M7 graph B failed");
    require(reg.submit_invocation(a,sa,ea,ga,ia,ra,53)==error::OK, "M7 submit failed");
    require(reg.extract(a,&snap)==error::OK && snap.graph_state==graph_phase::COMPLETE && snap.busy_device_count==0 && snap.invocation.value==0,
            "M7 mutation did not expose early-release shape");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,53,&ib)==error::OK, "M7 mutation did not release competing context early");
}

static void m7b() {
    Registry reg(test_mutation::M7_SUBMIT_RELEASES_DEVICES_EARLY); error err = error::OK; const auto a = reg.create_context(err), b = reg.create_context(err); require(err == error::OK, "M7b create failed");
    require(reg.bind_backend(a,2)==error::OK && reg.bind_backend(b,2)==error::OK, "M7b bind failed");
    SessionId sa{}, sb{}; SessionResetEpoch ea{}, eb{}; auto ra = root_token(92), rb = root_token(93);
    require(reg.attach_root(a,ra,&sa,&ea)==error::OK && reg.attach_root(b,rb,&sb,&eb)==error::OK, "M7b attach failed");
    GraphEpoch ga{}, gb{}; InvocationId ia{}, ib{}; const int d[] = {2}; const int p[] = {55}; snapshot snap{};
    require(reg.begin_graph(a,sa,ea,ra,&ga)==error::OK && reg.begin_invocation(a,sa,ea,ga,ra,d,1,p,1,55,&ia)==error::OK, "M7b invoke A failed");
    require(reg.begin_graph(b,sb,eb,rb,&gb)==error::OK, "M7b graph B failed");
    require(reg.submit_quarantined_invocation(a,sa,ea,ga,ia,ra,55)==error::OK, "M7b quarantine failed");
    require(reg.extract(a,&snap)==error::OK && snap.graph_state==graph_phase::QUARANTINED && snap.busy_device_count==0 && snap.invocation.value==0,
            "M7b mutation did not expose early-release quarantine shape");
    require(reg.begin_invocation(b,sb,eb,gb,rb,d,1,p,1,55,&ib)==error::OK, "M7b mutation did not release competing context early after quarantine");
}

int main(int argc, char ** argv) {
    const char * which = argc > 2 && std::strcmp(argv[1], "--case") == 0 ? argv[2] : "all";
    if (std::strcmp(which, "G5a") == 0) g5a();
    else if (std::strcmp(which, "G6") == 0) g6();
    else if (std::strcmp(which, "G6b") == 0) g6b();
    else if (std::strcmp(which, "G7") == 0) g7();
    else if (std::strcmp(which, "H8") == 0) h8();
    else if (std::strcmp(which, "H12") == 0) h12();
    else if (std::strcmp(which, "H12b") == 0) h12b();
    else if (std::strcmp(which, "H13b") == 0) h13b();
    else if (std::strcmp(which, "M7") == 0) m7();
    else if (std::strcmp(which, "M7b") == 0) m7b();
    else { g5a(); g6(); g6b(); g7(); h8(); h12(); h12b(); h13b(); m7(); m7b(); }
    return 0;
}
