// Compatibility entry: the backdrop, the sphere-layout guide, the key light, its shadows and the
// image-based lighting environment now live in one module, src/render/gfx/Environment.js.
// `Background` is that class under its old name (same constructor options { noise, guide }, same
// guide / guideOpacity / setColors API), so existing callers keep working unchanged.
export { Environment as Background, Environment, ENV_PRESETS } from './gfx/Environment.js';
