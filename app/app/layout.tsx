import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GACA Early Warning System",
  description: "Temperature forecasting dashboard for Southwestern Ontario",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
