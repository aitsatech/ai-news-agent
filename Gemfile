# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll", "~> 4.3.4"
gem "jekyll-theme-chirpy", "~> 7.0"

# Explicitly list these so the build runner can find them
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

gem "csv"
gem "base64"

group :test do
  gem "html-proofer", "~> 5.0"
end

platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]
