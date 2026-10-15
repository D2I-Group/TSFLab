import type { Metadata } from "next";
import { Inter, Newsreader, Fira_Code, Noto_Serif_SC } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const newsreader = Newsreader({
  subsets: ["latin"],
  variable: "--font-newsreader",
  display: "swap",
  style: ["normal", "italic"],
});
const firaCode = Fira_Code({ subsets: ["latin"], variable: "--font-fira-code", display: "swap" });
const notoSerifSC = Noto_Serif_SC({
  weight: ["400", "600", "700"],
  variable: "--font-noto-serif-sc",
  display: "swap",
  preload: false,
});

export const metadata: Metadata = {
  title: "TSFLab — Time-Series Forecasting Leaderboard · Diaugeia.AI",
  description:
    "An open, transparent leaderboard for time-series forecasting — covering static and real-time evaluation, with community submissions categorized and ranked across tracks, datasets, and horizons.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${newsreader.variable} ${firaCode.variable} ${notoSerifSC.variable}`}
    >
      {/* suppressHydrationWarning: theme class + browser extensions inject attrs */}
      <body className="flex min-h-screen flex-col" suppressHydrationWarning>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
