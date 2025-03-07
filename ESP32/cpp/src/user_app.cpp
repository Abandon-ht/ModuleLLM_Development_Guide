#include "user_app.h"
#include "lvgl_port_m5stack.hpp"

#define PUHUITI_FONT &AliBaBa_PuHuiTi_R_20
// #define PUHUITI_FONT LV_FONT_DEFAULT

lv_obj_t* background;
lv_obj_t* photoaction;
lv_obj_t* returnbutton;

lv_obj_t* initialization_state;
lv_obj_t* vllm_title;
lv_obj_t* vllm_output;

void vllm_display(void)
{
    if (lvgl_port_lock()) {
        background = lv_img_create(lv_scr_act());
        lv_img_set_src(background, &background_jpg);
        lv_obj_set_pos(background, 0, 0);

        photoaction = lv_img_create(lv_scr_act());
        lv_img_set_src(photoaction, &photoaction_jpg);
        lv_obj_set_pos(photoaction, 0, 0);
        lv_obj_add_flag(photoaction, LV_OBJ_FLAG_HIDDEN);

        returnbutton = lv_img_create(lv_scr_act());
        lv_img_set_src(returnbutton, &return_jpg);
        lv_obj_set_pos(returnbutton, 0, 0);
        lv_obj_add_flag(returnbutton, LV_OBJ_FLAG_HIDDEN);
        lv_obj_set_style_bg_opa(returnbutton, LV_OPA_70, LV_STATE_DEFAULT);
        lv_obj_add_flag(returnbutton, LV_OBJ_FLAG_HIDDEN);

        initialization_state = lv_label_create(lv_scr_act());
        lv_obj_set_style_text_font(initialization_state, PUHUITI_FONT, 0);
        lv_label_set_text(initialization_state, "LLM Module Loding ...");
        lv_obj_set_style_text_color(initialization_state, lv_color_hex(0x00ccff), 0);
        lv_obj_set_style_text_align(initialization_state, LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_align(initialization_state, LV_ALIGN_CENTER, 0, 0);
        lv_obj_set_style_bg_color(initialization_state, lv_color_white(), 0);

        vllm_title = lv_label_create(lv_scr_act());
        lv_obj_set_style_text_font(vllm_title, PUHUITI_FONT, 0);
        lv_obj_set_style_text_color(vllm_title, lv_color_white(), 0);
        lv_label_set_text(vllm_title, "VLLM Output");
        lv_obj_align(vllm_title, LV_ALIGN_TOP_MID, 0, 0);
        lv_obj_add_flag(vllm_title, LV_OBJ_FLAG_HIDDEN);

        vllm_output = lv_textarea_create(lv_scr_act());
        lv_obj_set_style_text_font(vllm_output, PUHUITI_FONT, 0);
        lv_obj_set_style_text_color(vllm_output, lv_color_white(), 0);
        lv_obj_set_style_border_width(vllm_output, 2, 0);
        lv_obj_set_style_border_color(vllm_output, lv_color_hex(0x00ccff), 0);
        lv_obj_set_style_bg_color(vllm_output, lv_color_black(), 0);
        lv_obj_set_style_text_align(vllm_output, LV_TEXT_ALIGN_CENTER, 0);
        lv_textarea_set_max_length(vllm_output, 1024 * 10);
        lv_textarea_set_text(vllm_output, "");
        lv_obj_set_size(vllm_output, 300, 200);
        lv_obj_align(vllm_output, LV_ALIGN_OUT_TOP_LEFT, 10, 30);
        lv_obj_add_flag(vllm_output, LV_OBJ_FLAG_HIDDEN);
        lv_obj_set_style_bg_color(vllm_output, lv_color_black(), 0);


        lvgl_port_unlock();
    }
}
