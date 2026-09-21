/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        // Light operations theme. Saturation on this dashboard carries meaning:
        // green/amber/orange/red are the damage ladder and nothing else, so the
        // interactive accent is blue and the surfaces stay neutral.
        diq: {
          navy: "#1B2A4A", // headings and primary ink
          blue: "#1D4ED8", // interactive accent: buttons, focus, active state
          orange: "#C2410C", // damage: major (text-safe on white)
          red: "#B91C1C", // damage: destroyed
          bg: "#F4F6F9", // page ground
          panel: "#FFFFFF", // panel surfaces
          line: "#D8DEE9", // hairline borders and rules
          ink: "#0F1B2D", // strongest text
          muted: "#5B6B85", // secondary text
        },
      },
      fontFamily: {
        label: [
          "ui-monospace",
          "JetBrains Mono",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
