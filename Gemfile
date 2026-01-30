# frozen_string_literal: true

source "https://rubygems.org"

# Use Jekyll 4 for better performance and 2026 compatibility
gem "jekyll", "~> 4.4"

# The core theme gem
gem "jekyll-theme-chirpy", "~> 7.1"

# REQUIRED PLUGINS for GitHub Pages + Chirpy
# These must match the 'plugins' list in your _config.yml
group :jekyll_plugins do
  gem "jekyll-remote-theme"
  gem "jekyll-include-cache"
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
end

# Testing and local development tools
gem "html-proofer", "~> 5.0", group: :test

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
