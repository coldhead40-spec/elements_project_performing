#!/bin/sh
# Re-encode the 4 WhatsApp clips to HAP so TouchDesigner can loop/scrub them smoothly on the GPU.
# (WhatsApp H.264 is long-GOP: random access and 4 simultaneous streams can stutter.)
# If your ffmpeg has no HAP encoder, record them with TouchDesigner's Movie File Out TOP (codec: HAP).
#   13_40_47 = AIR   13_40_48 = EARTH (flowers)   13_40_49 = FIRE   13_40_49_1_ = WATER
ffmpeg -i WhatsApp_Video_2026-11-19_at_13_40_47.mp4     -an -c:v hap -format hap air.mov
ffmpeg -i WhatsApp_Video_2026-11-19_at_13_40_48.mp4     -an -c:v hap -format hap earth.mov
ffmpeg -i WhatsApp_Video_2026-11-19_at_13_40_49.mp4     -an -c:v hap -format hap fire.mov
ffmpeg -i WhatsApp_Video_2026-11-19_at_13_40_49_1_.mp4  -an -c:v hap -format hap water.mov
# Grab frame 0 as the painting base (or use the high-quality image from your brief instead):
ffmpeg -i WhatsApp_Video_2026-11-19_at_13_40_47.mp4 -frames:v 1 base_frame.png
