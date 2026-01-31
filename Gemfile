# frozen_string_literal: true

source "https://rubygems.org"

# Core Jekyll and Chirpy Theme
gem "jekyll", "~> 4.3.4"
gem "jekyll-theme-chirpy", "~> 7.0"

# Explicitly list all plugins from your _config.yml
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
  gem "jekyll-include-cache"
  gem "jekyll-archives"
  gem "jekyll-redirect-from"
  gem "jekyll-twitter-plugin"
end

# Required for Ruby 3.3+ compatibility
gem "csv"
gem "base64"
gem "webrick" # Required for local serving in newer Ruby versions

# Support for different Operating Systems
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]
