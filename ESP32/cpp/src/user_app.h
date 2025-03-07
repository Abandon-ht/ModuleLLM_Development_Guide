#pragma once

#include "lvgl.h"

LV_FONT_DECLARE(AliBaBa_PuHuiTi_R_20);

LV_IMG_DECLARE(background_jpg)
LV_IMG_DECLARE(photoaction_jpg)
LV_IMG_DECLARE(return_jpg)

extern lv_obj_t* background;
extern lv_obj_t* photoaction;
extern lv_obj_t* returnbutton;

extern lv_obj_t* initialization_state;
extern lv_obj_t* vllm_title;
extern lv_obj_t* vllm_output;