// Host test, loads no model: common_chat_templates_need_jinja() says whether the built-in (non-jinja) chat template
// matcher can render a template set. llama-completion runs the built-in matcher by default and has to switch to the
// jinja engine for a model whose template the matcher does not know.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "chat.h"

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

// A template no entry of the built-in matcher recognises.
static const char * UNKNOWN_TEMPLATE =
    "{% for m in messages %}<<turn:{{ m.role }}>>{{ m.content }}<<end>>{% endfor %}";

static common_chat_templates_inputs one_user_message(bool use_jinja) {
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "hello gemma";

    common_chat_templates_inputs inputs;
    inputs.use_jinja             = use_jinja;
    inputs.messages              = { msg };
    inputs.add_generation_prompt = true;
    return inputs;
}

int main() {
    // a template the built-in matcher knows: the "chatml" override resolves to the ChatML source
    {
        auto tmpls = common_chat_templates_init(/* model= */ nullptr, "chatml");
        CHECK(!common_chat_templates_need_jinja(tmpls.get()));
    }

    // a literal ChatML template text is known too
    {
        const std::string chatml =
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
            "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
        auto tmpls = common_chat_templates_init(/* model= */ nullptr, chatml);
        CHECK(!common_chat_templates_need_jinja(tmpls.get()));
    }

    // a template it does not know needs the jinja engine
    {
        auto tmpls = common_chat_templates_init(/* model= */ nullptr, UNKNOWN_TEMPLATE);
        CHECK(common_chat_templates_need_jinja(tmpls.get()));

        // the jinja engine renders it
        const auto prompt = common_chat_templates_apply(tmpls.get(), one_user_message(true)).prompt;
        CHECK(prompt.find("hello gemma") != std::string::npos);
        CHECK(prompt.find("<<turn:user>>") != std::string::npos);

        // the built-in matcher refuses it, and says to use --jinja: this is why the fallback exists
        bool threw = false;
        try {
            common_chat_templates_apply(tmpls.get(), one_user_message(false));
        } catch (const std::runtime_error & e) {
            threw = std::string(e.what()).find("try using --jinja") != std::string::npos;
        }
        CHECK(threw);
    }

    return 0;
}
