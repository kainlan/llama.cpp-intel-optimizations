# llama.cpp-9wjb evidence — 2026-08-08

test-backend-ops -o SET_ROWS -b SYCL0 at 0b73dfdad, B70 level_zero:0, rc=134.
Assert: set_rows.cpp:1064 GGML_ASSERT(src0->type == GGML_TYPE_F32) — pre-existing
(verbatim at a5008abf9:961, introduced by aa2e19fa5), surfaced by the nhip battery's
regression leg. Full stderr incl. gdb backtrace in setrows-reg-abort.txt (.txt because
.gitignore drops *.log). ROPE_SET_ROWS same run: rc=0.
