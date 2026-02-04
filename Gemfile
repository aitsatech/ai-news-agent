# frozen_string_literal: true

source "https://rubygems.org"

# Apex Jekyll Setup
gem "jekyll", "~> 4.3.4"
gem "jekyll-theme-chirpy", "~> 7.0.1"

# Essential Plugins for Chirpy Functionality
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
  gem "jekyll-include-cache"
  gem "jekyll-archives"
  gem "jekyll-redirect-from"
end

# Runtime Compatibility for Ruby 3.3 (Ensures GitHub Actions stability)
gem "webrick"
gem "base64"
gem "csv"
gem "bigdecimal"

# Optimization: Handles high-performance asset compression
gem "unicode-display_width", "~> 2.5"
gem "terminal-table", "~> 3.0"

# Performance & Environment handling
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]
