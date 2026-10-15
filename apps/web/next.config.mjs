/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export to `out/` — served as plain files by HF Space (static SDK)
  // and by Cloudflare Pages. Both serve at the domain root, so no basePath.
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
