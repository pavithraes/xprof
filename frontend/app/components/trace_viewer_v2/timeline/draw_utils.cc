#include "xprof/frontend/app/components/trace_viewer_v2/timeline/draw_utils.h"

#include <algorithm>
#include <cmath>

#include "xprof/frontend/app/components/trace_viewer_v2/color/colors.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

namespace {

// GM3-style indefinite linear progress indicator constants.
// The width of the progress bar as a ratio of the viewport width.
constexpr float kProgressBarWidthRatio = 0.4f;
constexpr Pixel kProgressBarHeight = 4.0f;
// The duration in seconds for one full animation cycle of the progress bar.
constexpr float kAnimationDuration = 3.0f;
// The gap in pixels between the primary and secondary indicators.
constexpr Pixel kProgressBarGap = 4.0f;
// A scaling factor used in easing calculations for animation.
constexpr float kProgressScale = 2.0f;
constexpr Pixel kTextOffsetY = 16.0f;

// Draws a "Loading data..." text message centered below the progress bar.
void DrawLoadingText(const ImGuiViewport* viewport) {
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  if (!draw_list) return;

  const char* text = "Loading data...";
  const ImVec2 text_size = ImGui::CalcTextSize(text);
  const ImVec2 center = viewport->GetCenter();

  const float text_x = center.x - text_size.x / 2.0f;
  // Position text below the progress bar. The progress bar's center is at
  // viewport->GetCenter().y.
  const float text_y = center.y + kProgressBarHeight / 2.0f + kTextOffsetY;

  draw_list->AddText(ImVec2(text_x, text_y), kBlackColor, text);
}

}  // namespace

// Draws a loading indicator in the center of the viewport.
void DrawLoadingIndicator(const ImGuiViewport* viewport) {
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  if (!draw_list) return;

  const Pixel kProgressBarCornerRadius = kProgressBarHeight / 2.0f;
  const Pixel progress_bar_width = viewport->Size.x * kProgressBarWidthRatio;

  const ImVec2 center = viewport->GetCenter();
  const Pixel start_x = center.x - progress_bar_width / 2.0f;
  const Pixel end_x = center.x + progress_bar_width / 2.0f;
  const Pixel y = center.y - kProgressBarHeight / 2.0f;

  const float time = static_cast<float>(ImGui::GetTime());
  const float cycle_progress =
      fmod(time, kAnimationDuration) / kAnimationDuration;

  // The animation of the primary progress bar is defined by the movement of its
  // head (right side) and tail (left side). The head moves with an ease-out
  // curve, starting fast and slowing down, while the tail moves linearly. This
  // creates the effect of the bar first growing in width and then shrinking as
  // the tail catches up to the head.

  // Quadratic ease-out for the head of the bar.
  const float head_progress =
      kProgressScale * cycle_progress * (kProgressScale - cycle_progress);
  // Linear progress for the tail of the bar.
  const float tail_progress = kProgressScale * cycle_progress;

  const Pixel primary_start = start_x + progress_bar_width * tail_progress;
  const Pixel primary_end = start_x + progress_bar_width * head_progress;

  // Draw the two secondary segments and one primary segment, with gaps.
  // All segments are clipped to the progress bar's bounds.
  // Something like this:
  // (-----s1-----) (--p--) (-----s2-----)
  //               ^  gap  ^
  const Pixel s1_start = start_x;
  const Pixel s1_end = std::min(primary_start - kProgressBarGap, end_x);
  if (s1_start < s1_end) {
    draw_list->AddRectFilled(
        ImVec2(s1_start, y), ImVec2(s1_end, y + kProgressBarHeight),
        kSecondary90, kProgressBarCornerRadius, ImDrawFlags_RoundCornersAll);
  }

  const Pixel p_start = std::max(primary_start, start_x);
  const Pixel p_end = std::min(primary_end, end_x);
  if (p_start < p_end) {
    draw_list->AddRectFilled(
        ImVec2(p_start, y), ImVec2(p_end, y + kProgressBarHeight), kPrimary40,
        kProgressBarCornerRadius, ImDrawFlags_RoundCornersAll);
  }

  const Pixel s2_start = std::max(primary_end + kProgressBarGap, start_x);
  const Pixel s2_end = end_x;
  if (s2_start < s2_end) {
    draw_list->AddRectFilled(
        ImVec2(s2_start, y), ImVec2(s2_end, y + kProgressBarHeight),
        kSecondary90, kProgressBarCornerRadius, ImDrawFlags_RoundCornersAll);
  }

  DrawLoadingText(viewport);
}

}  // namespace traceviewer
