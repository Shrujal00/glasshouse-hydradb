# FRAME.MD — Glasshouse Hyperframes Video Design System

> Hyperframes (by HeyGen) video composition design specification for Glasshouse Enterprise Context Engine.

## Composition Meta
- **Composition ID**: `glasshouse-hero-demo`
- **Resolution**: `1920x1080` (16:9 4K-scalable canvas)
- **FPS**: `60`
- **Duration**: `8 seconds` (480 frames)

## Color Palette
- **Background**: `#070709` (Obsidian Void)
- **Primary Accent**: `#FF5C39` (Neon Orange Glow)
- **Secondary Accent**: `#FF7A00` (Amber Prior)
- **Purple Accent**: `#A855F7` (ADR / Document Node)
- **Typography**: `#F3F4F6` (Main), `#9CA3AF` (Muted)

## Tracks & Timeline Structure
- **Track 0**: Background Grid & Ambient Spotlight (0s - 8s)
- **Track 1**: Title Overlay & Hero Motion Graphics (0s - 3s)
- **Track 2**: Graph Traversal Multi-Hop Sequence (2s - 6s)
- **Track 3**: Resolved Entity Glass Tooltip & Citation Callout (5s - 8s)

## GSAP Binding Schema
```javascript
const tl = gsap.timeline({ paused: true });
tl.from("#video-title", { opacity: 0, y: 50, duration: 1 }, 0);
tl.to("#camera-stage", { scale: 1.4, x: -100, duration: 2, ease: "power2.inOut" }, 3);
window.__timelines = window.__timelines || {};
window.__timelines["glasshouse-hero-demo"] = tl;
```
