import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver, but Recharts' ResponsiveContainer requires one.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as any;
}
