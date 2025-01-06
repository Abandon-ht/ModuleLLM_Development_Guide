/*
 * MIT License
 *
 * Copyright (c) 2024 LittleMouse
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * - The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "image.h"
#include "audio.h"
#include "M5CoreS3.h"
#include <M5ModuleLLM.h>

#include <ArduinoJson.h>
#include <FreeRTOS.h>

#define Serial USBSerial  // PlatformIO

M5Canvas canvas(&M5.Display);

SemaphoreHandle_t mutex;

extern const uint16_t rgb565[];
extern const uint8_t wav_unsigned_8bit_click[46000];
extern const uint8_t wav_shutter_data[19584];

struct box_t {
    int x;
    int y;
    int w;
    int h;
    std::uint16_t color;
    int touch_id = -1;

    void clear(void)
    {
        M5.Display.setColor(M5.Display.getBaseColor());
        for (int i = 0; i < 8; ++i) {
            M5.Display.drawRoundRect(x + i, y + i, w - i * 2, h - i * 2, 20);
        }
    }
    void draw(void)
    {
        canvas.setColor(color);
        int ie = touch_id < 0 ? 4 : 8;
        for (int i = 0; i < ie; ++i) {
            canvas.drawRoundRect(x + i, y + i, w - i * 2, h - i * 2, 20);
        }
    }
    bool contain(int x, int y)
    {
        return this->x <= x && x < (this->x + this->w) && this->y <= y && y < (this->y + this->h);
    }
};

struct KeyPoint {
    float x;
    float y;
    float c;
};

struct yolo_box_t {
    String class_name;
    String model;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
    int frame;
    KeyPoint keypoint;
};

struct VlmData {
    String delta;
    bool finish;
};

const int BlueButtonPin                = 8;
const int RedButtonPin                 = 9;
static constexpr std::size_t box_count = 7;
static box_t box_list[box_count];
static yolo_box_t yolo_box;
static VlmData vlm_data;
static yolo_box_t pose_boxes[17];
static yolo_box_t hand_boxes[21];
static float masks[32];
static int state      = 0;
static bool button    = false;
static bool inference = false;
static M5ModuleLLM module_llm;
// static String camera_work_id;
static String yolo_work_id;
static String vlm_work_id;
static String melotts_work_id;

const size_t JSON_BUFFER_SIZE = 2048;
char jsonBuffer[JSON_BUFFER_SIZE];
size_t bufferIndex = 0;

void play_wav()
{
    CoreS3.Speaker.playRaw(wav_unsigned_8bit_click,
                           sizeof(wav_unsigned_8bit_click) / sizeof(wav_unsigned_8bit_click[0]), 44100, false);
}

void play_camera_wav()
{
    CoreS3.Speaker.playRaw(wav_shutter_data, sizeof(wav_shutter_data) / sizeof(wav_shutter_data[0]), 44100, false);
}

void setup_bsp(void)
{
    pinMode(BlueButtonPin, INPUT_PULLUP);
    pinMode(RedButtonPin, INPUT_PULLUP);
}

void setup_lcd(void)
{
    canvas.setColorDepth(16);
    canvas.setFont(&fonts::lgfxJapanMinchoP_32);
    canvas.setTextWrap(false);
    canvas.createSprite(M5.Display.width(), M5.Display.height());
}

void setup_menu(void)
{
    M5.Display.pushImage(0, 0, CoreS3.Display.width(), CoreS3.Display.height(), rgb565);

    M5.Display.setTextDatum(middle_center);
    M5.Display.setFont(&fonts::Orbitron_Light_24);

    for (std::size_t i = 0; i < box_count - 1; ++i) {
        box_list[i].w = 100;
        box_list[i].h = 100;
    }
    box_list[0].x = 35;
    box_list[0].y = 12;
    box_list[1].x = 187;
    box_list[1].y = 12;
    box_list[2].x = 35;
    box_list[2].y = 130;
    box_list[3].x = 187;
    box_list[3].y = 130;

    box_list[4].x = 135;
    box_list[4].y = 130;
    box_list[4].w = 50;
    box_list[4].h = 100;
}

void setup_camera(void)
{
    CoreS3.Display.setTextColor(ORANGE);
    CoreS3.Display.setTextDatum(middle_center);
    CoreS3.Display.setFont(&fonts::Orbitron_Light_24);

    if (!CoreS3.Camera.begin()) {
        CoreS3.Display.drawString("Camera Init Fail", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    }
    CoreS3.Display.drawString("Camera Init Success", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    CoreS3.Camera.sensor->set_framesize(CoreS3.Camera.sensor, FRAMESIZE_QVGA);
}

void setup_comm(void)
{
    Serial.begin(115200);
    Serial2.setRxBufferSize(2048);
    Serial2.begin(1500000, SERIAL_8N1, 18, 17);
}

void setup_llm(void)
{
    M5.Display.setTextColor(ORANGE);
    M5.Display.setTextSize(1);
    M5.Display.setTextDatum(middle_center);
    module_llm.begin(&Serial2);
    M5.Display.drawString("ModuleLLM is connecting..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    while (1) {
        if (module_llm.checkConnection()) {
            M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
            break;
        }
    }

    M5.Display.drawString("Reset ModuleLLM..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    module_llm.sys.reset();
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("Setup ok..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
}

void setup_yolo_detect(void)
{
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("setup_yolo_detect..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    m5_module_llm::ApiYoloSetupConfig_t yolo_config;
    yolo_config.input = {"yolo.jpeg.base64"};
    yolo_config.model = "yolo11n";
    yolo_box.model    = "yolo11n";
    yolo_work_id      = module_llm.yolo.setup(yolo_config, "yolo_setup");
    while (yolo_work_id == nullptr) vTaskDelay(100);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("yolo setup done..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    vTaskDelay(200);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
}

void setup_yolo_pose(void)
{
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("setup_yolo_pose..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    m5_module_llm::ApiYoloSetupConfig_t yolo_config;
    yolo_config.input = {"yolo.jpeg.base64"};
    yolo_config.model = "yolo11n-pose";
    yolo_box.model    = "yolo11n-pose";
    yolo_work_id      = module_llm.yolo.setup(yolo_config, "yolo_setup");
    while (yolo_work_id == nullptr) vTaskDelay(100);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("yolo setup done..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    vTaskDelay(200);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
}

void setup_yolo_seg(void)
{
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("setup_yolo_seg..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    m5_module_llm::ApiYoloSetupConfig_t yolo_config;
    yolo_config.input = {"yolo.jpeg.base64"};
    yolo_config.model = "yolo11n-seg";
    yolo_box.model    = "yolo11n-seg";
    yolo_work_id      = module_llm.yolo.setup(yolo_config, "yolo_setup");
    while (yolo_work_id == nullptr) vTaskDelay(100);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("yolo setup done..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    vTaskDelay(200);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
}

void setup_yolo_hand(void)
{
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("setup_yolo_hand..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    m5_module_llm::ApiYoloSetupConfig_t yolo_config;
    yolo_config.input = {"yolo.jpeg.base64"};
    yolo_config.model = "yolo11n-hand-pose";
    yolo_box.model    = "yolo11n-hand-pose";
    yolo_work_id      = module_llm.yolo.setup(yolo_config, "yolo_setup");
    while (yolo_work_id == nullptr) vTaskDelay(100);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("yolo setup done..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    vTaskDelay(200);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
}

void setup_vlm(void)
{
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("setup_vlm..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    m5_module_llm::ApiVlmSetupConfig_t vlm_config;
    vlm_work_id = module_llm.vlm.setup(vlm_config, "vlm_setup");
    while (vlm_work_id == nullptr) vTaskDelay(100);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("vlm setup done..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    vTaskDelay(200);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
}

void setup_melotts(void)
{
    M5.Display.drawString("setup_tts..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    m5_module_llm::ApiMelottsSetupConfig_t melotts_config;
    melotts_config.input = {vlm_work_id};
    melotts_work_id      = module_llm.melotts.setup(melotts_config, "melotts_setup");
    while (melotts_work_id == nullptr) vTaskDelay(100);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.drawString("tts setup done..", CoreS3.Display.width() / 2, CoreS3.Display.height() / 2);
    vTaskDelay(200);
    M5.Display.fillRect(0, (CoreS3.Display.height() / 2) - 10, 320, 25, WHITE);
    M5.Display.setFont(&fonts::efontCN_24);
}

void setup_yolo(int j)
{
    void (*setupFunctions[4])() = {setup_yolo_detect, setup_yolo_pose, setup_yolo_hand, setup_vlm};

    if (j >= 0 && j < 3) {
        setupFunctions[j]();
        state = 1;
    }

    if (j == 3) {
        setupFunctions[j]();
        setup_melotts();
        state = 2;
    }
}

void menuTask(void* pvParameters)
{
    while (true) {
        if (state) {
            vTaskDelay(100);
            continue;
        }

        M5.delay(1);
        M5.update();

        auto count = M5.Touch.getCount();
        if (!count) {
            vTaskDelay(10);
            continue;
        }

        static m5::touch_state_t prev_state;
        auto t = M5.Touch.getDetail();
        if (prev_state != t.state) {
            prev_state                                  = t.state;
            static constexpr const char* state_name[16] = {
                "none", "touch", "touch_end", "touch_begin", "___", "hold", "hold_end", "hold_begin",
                "___",  "flick", "flick_end", "flick_begin", "___", "drag", "drag_end", "drag_begin"};
            M5_LOGI("%s", state_name[t.state]);
        }

        for (std::size_t i = 0; i < count; ++i) {
            auto t = M5.Touch.getDetail(i);

            for (std::size_t j = 0; j < box_count - 3; ++j) {
                if (t.wasClicked() && box_list[j].contain(t.x, t.y)) {
                    play_wav();
                    box_list[j].touch_id = t.id;
                    setup_yolo(j);
                }
                if (box_list[j].touch_id == t.id) {
                    if (t.wasReleased()) {
                        box_list[j].touch_id = -1;
                    }
                }
            }
        }
        vTaskDelay(10);
    }
}

void menuBackTask(void* pvParameters)
{
    while (true) {
        if (!state) {
            vTaskDelay(100);
            continue;
        }

        M5.update();

        box_list[5].x     = 0;
        box_list[5].y     = 0;
        box_list[5].w     = 100;
        box_list[5].h     = 70;
        box_list[5].color = TFT_WHITE;

        box_list[6].x     = 0;
        box_list[6].y     = 0;
        box_list[6].w     = 320;
        box_list[6].h     = 240;
        box_list[6].color = TFT_ORANGE;

        auto count = M5.Touch.getCount();

        for (std::size_t i = 0; i < count; ++i) {
            auto t = M5.Touch.getDetail(i);
            if (t.wasClicked() && box_list[5].contain(t.x, t.y)) {
                play_wav();
                box_list[5].touch_id = t.id;
                state                = 0;
                vTaskDelay(100);
                if (!yolo_work_id.isEmpty()) module_llm.yolo.exit(yolo_work_id);
                if (!vlm_work_id.isEmpty()) module_llm.vlm.exit(vlm_work_id);
                if (!melotts_work_id.isEmpty()) module_llm.melotts.exit(melotts_work_id);
                setup_menu();
            }

            if (box_list[5].touch_id == t.id) {
                if (t.wasReleased()) {
                    box_list[5].touch_id = -1;
                }
            }
        }

        vTaskDelay(100);
    }
}

void send_camera_data(uint8_t* out_jpg, size_t out_jpg_len, String work_id)
{
    JsonDocument jsonDoc;
    jsonDoc["RAW"]        = out_jpg_len;
    jsonDoc["request_id"] = "yolo_inference";
    jsonDoc["work_id"]    = work_id;
    jsonDoc["action"]     = "inference";
    jsonDoc["object"]     = "cv.jpeg.base64";

    char jsonString[256];
    serializeJson(jsonDoc, jsonString);
    Serial2.write((const uint8_t*)jsonString, strlen(jsonString));
    Serial2.write(out_jpg, out_jpg_len);
}

void parseJson(const char* jsonString)
{
    JsonDocument jsonDoc;
    // Serial.println(jsonString);
    DeserializationError error = deserializeJson(jsonDoc, jsonString);
    if (error) return;

    if (jsonDoc["request_id"].isNull()) return;
    const char* request_id = jsonDoc["request_id"];

    if (strcmp(request_id, "yolo_inference") == 0) {
        JsonArray delta = jsonDoc["data"]["delta"].as<JsonArray>();

        if (delta.size() > 0) {
            JsonObject result   = delta[0].as<JsonObject>();
            yolo_box.class_name = result["class"].as<String>();
            yolo_box.confidence = result["confidence"].as<float>();
            JsonArray bboxArray = result["bbox"].as<JsonArray>();

            if (bboxArray.size() == 4) {
                yolo_box.frame = 1;
                yolo_box.x1    = bboxArray[0].as<float>();
                yolo_box.y1    = bboxArray[1].as<float>();
                yolo_box.x2    = bboxArray[2].as<float>();
                yolo_box.y2    = bboxArray[3].as<float>();
            }

            JsonArray kps = result["kps"].as<JsonArray>();

            if (yolo_box.model == "yolo11n-pose") {
                if (kps.size() == 51) {
                    for (int i = 0; i < 17; ++i) {
                        pose_boxes[i].keypoint.x = kps[i * 3].as<float>();
                        pose_boxes[i].keypoint.y = kps[i * 3 + 1].as<float>();
                        pose_boxes[i].keypoint.c = kps[i * 3 + 2].as<float>();
                    }
                }
            }

            if (yolo_box.model == "yolo11n-hand-pose") {
                if (kps.size() == 63) {
                    for (int i = 0; i < 21; ++i) {
                        hand_boxes[i].keypoint.x = kps[i * 3].as<float>();
                        hand_boxes[i].keypoint.y = kps[i * 3 + 1].as<float>();
                        hand_boxes[i].keypoint.c = kps[i * 3 + 2].as<float>();
                    }
                }
            }

            JsonArray mask = result["mask"].as<JsonArray>();

            if (yolo_box.model == "yolo11n-mask") {
                if (kps.size() == 32) {
                    for (int i = 0; i < 32; ++i) {
                        masks[i] = mask[i].as<float>();
                    }
                }
            }
        }
    }

    if (strcmp(request_id, "vlm_inference") == 0) {
        vlm_data.delta  = jsonDoc["data"]["delta"].as<String>();
        vlm_data.finish = jsonDoc["data"]["finish"].as<bool>();
        jsonDoc.clear();
        M5.Display.print(vlm_data.delta.c_str());
        if (vlm_data.finish) {
            vTaskDelay(8000);
            inference = false;
        }
    }
}

void recv_llm_data(void)
{
    static int bufferIndex = 0;

    while (Serial2.available() > 0) {
        char c = Serial2.read();

        if (c == '\n') {
            if (bufferIndex > 0) {
                jsonBuffer[bufferIndex] = '\0';
                parseJson(jsonBuffer);
                bufferIndex = 0;
            }
        } else if (bufferIndex < JSON_BUFFER_SIZE - 1) {
            jsonBuffer[bufferIndex++] = c;
        } else {
            bufferIndex = 0;
        }
    }
}

void recvTask(void* pvParameters)
{
    while (true) {
        if (state) {
            recv_llm_data();
        } else {
            vTaskDelay(100);
        }
    }
}

void button_task(void* pvParameters)
{
    while (true) {
        if (!digitalRead(BlueButtonPin)) {
            play_camera_wav();
            button = true;
            vTaskDelay(1000);
        }
        if (!digitalRead(RedButtonPin)) {
            play_wav();
            state = 0;
            vTaskDelay(100);
            if (!yolo_work_id.isEmpty()) module_llm.yolo.exit(yolo_work_id);
            if (!vlm_work_id.isEmpty()) module_llm.vlm.exit(vlm_work_id);
            if (!melotts_work_id.isEmpty()) module_llm.melotts.exit(melotts_work_id);
            setup_menu();
        }
        vTaskDelay(100);
    }
}

void cameraTask(void* pvParameters)
{
    while (true) {
        if (state == 1) {
            if (CoreS3.Camera.get()) {
                uint8_t* out_jpg   = NULL;
                size_t out_jpg_len = 0;
                frame2jpg(CoreS3.Camera.fb, 50, &out_jpg, &out_jpg_len);
                send_camera_data(out_jpg, out_jpg_len, yolo_work_id);
                free(out_jpg);
                canvas.pushImage(0, 0, CoreS3.Display.width(), CoreS3.Display.height(),
                                 (uint16_t*)CoreS3.Camera.fb->buf);
                canvas.drawString("<", 30, 40);
                if (yolo_box.frame) {
                    yolo_box.frame--;

                    canvas.setTextDatum(bottom_left);

                    canvas.drawString(yolo_box.class_name.c_str(), yolo_box.x1, yolo_box.y1 - 40);
                    canvas.drawFloat(yolo_box.confidence, 2, yolo_box.x2, yolo_box.y1 - 40);

                    canvas.drawRect(yolo_box.x1, yolo_box.y1 - 40, yolo_box.x2, yolo_box.y2 - 40, ORANGE);

                    const int pose_lines[][3] = {
                        {0, 2, NAVY},      {2, 4, DARKGREEN}, {0, 1, DARKCYAN},   {1, 3, MAROON},
                        {6, 5, PURPLE},    {6, 8, OLIVE},     {8, 10, LIGHTGREY}, {5, 7, DARKGREY},
                        {7, 9, BLUE},      {12, 11, GREEN},   {6, 12, CYAN},      {12, 14, RED},
                        {14, 16, MAGENTA}, {5, 11, YELLOW},   {11, 13, ORANGE},   {13, 15, GREENYELLOW}};

                    const int hand_lines[][3] = {{0, 1, NAVY},         {1, 2, DARKGREEN},     {2, 3, DARKCYAN},
                                                 {3, 4, MAROON},       {0, 5, PURPLE},        {5, 6, OLIVE},
                                                 {6, 7, LIGHTGREY},    {7, 8, DARKGREY},      {0, 17, BLUE},
                                                 {17, 18, GREEN},      {18, 19, CYAN},        {19, 20, RED},
                                                 {5, 9, MAGENTA},      {9, 13, YELLOW},       {13, 17, ORANGE},
                                                 {9, 10, GREENYELLOW}, {10, 11, MAGENTA},     {11, 12, YELLOW},
                                                 {13, 14, ORANGE},     {14, 15, GREENYELLOW}, {15, 16, GREENYELLOW}};

                    if (yolo_box.model == "yolo11n-pose") {
                        for (const auto& line : pose_lines) {
                            canvas.drawLine(pose_boxes[line[0]].keypoint.x, pose_boxes[line[0]].keypoint.y - 40,
                                            pose_boxes[line[1]].keypoint.x, pose_boxes[line[1]].keypoint.y - 40,
                                            line[2]);
                        }
                    }

                    if (yolo_box.model == "yolo11n-hand-pose") {
                        for (const auto& line : hand_lines) {
                            canvas.drawLine(hand_boxes[line[0]].keypoint.x, hand_boxes[line[0]].keypoint.y - 40,
                                            hand_boxes[line[1]].keypoint.x, hand_boxes[line[1]].keypoint.y - 40,
                                            line[2]);
                        }
                    }
                }

                M5.Display.startWrite();
                canvas.pushSprite(&M5.Display, 0, 0);
                M5.Display.endWrite();

                CoreS3.Camera.free();
            }
        } else if (state == 2) {
            if (!inference) {
                if (CoreS3.Camera.get()) {
                    uint8_t* out_jpg   = NULL;
                    size_t out_jpg_len = 0;
                    frame2jpg(CoreS3.Camera.fb, 50, &out_jpg, &out_jpg_len);
                    send_camera_data(out_jpg, out_jpg_len, vlm_work_id);
                    free(out_jpg);
                    canvas.pushImage(0, 0, CoreS3.Display.width(), CoreS3.Display.height(),
                                     (uint16_t*)CoreS3.Camera.fb->buf);
                    canvas.drawString("<", 30, 40);
                    if (button) {
                        box_list[6].draw();
                        module_llm.vlm.inference(vlm_work_id, "请用幽默的方式描述这张图片，字数不超过60个。");
                        button          = false;
                        inference       = true;
                        vlm_data.finish = false;
                        canvas.fillRect(0, 0, 320, 240, WHITE);
                        M5.Display.setCursor(50, 50);
                        M5.Display.setTextScroll(true);
                    }

                    M5.Display.startWrite();
                    canvas.pushSprite(&M5.Display, 0, 0);
                    M5.Display.endWrite();

                    CoreS3.Camera.free();
                } else {
                    vTaskDelay(100);
                }
            } else {
                vTaskDelay(100);
            }
        } else {
            vTaskDelay(100);
        }
    }
}

void setup_task(void)
{
    mutex = xSemaphoreCreateMutex();
    xTaskCreatePinnedToCore(recvTask, "Receive Task", 8192, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(cameraTask, "Camera Task", 8192, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(menuTask, "Menu Task", 8192, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(menuBackTask, "Menu Back Task", 8192, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(button_task, "Button Task", 8192, NULL, 3, NULL, 1);
}

void setup()
{
    CoreS3.begin();
    setup_bsp();
    setup_lcd();
    setup_camera();
    setup_menu();
    setup_comm();
    setup_llm();
    setup_task();
}

void loop()
{
}
