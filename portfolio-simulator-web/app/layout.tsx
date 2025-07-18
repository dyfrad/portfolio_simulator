import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Portfolio Simulator - Advanced Monte Carlo Analysis",
  description: "Run sophisticated Monte Carlo simulations to understand your portfolio's potential performance. Optimize asset allocation, test different scenarios, and make data-driven investment decisions.",
  keywords: ["portfolio", "simulation", "monte carlo", "investment", "optimization", "financial planning"],
  authors: [{ name: "Portfolio Simulator Team" }],
  viewport: "width=device-width, initial-scale=1",
  robots: "index, follow",
  openGraph: {
    title: "Portfolio Simulator - Advanced Monte Carlo Analysis",
    description: "Run sophisticated Monte Carlo simulations to understand your portfolio's potential performance.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "Portfolio Simulator - Advanced Monte Carlo Analysis",
    description: "Run sophisticated Monte Carlo simulations to understand your portfolio's potential performance.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
