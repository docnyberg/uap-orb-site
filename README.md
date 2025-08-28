# UAP / Orb Atlas — Netlify Skeleton (Full Viewer)

This skeleton lets you deploy your **full viewer** with a **visitor counter** on Netlify.

## Structure
```
uap-orb-site/
  public/
    index.html         # your full viewer (already inserted)
    atlas.csv          # place your atlas here
    thumbs/            # scene thumbs
    thumbs_obj/        # object thumbs
    svgs/              # svg files
  netlify/
    functions/
      unique.mjs       # visitor counter serverless function
  netlify.toml         # publish dir + /api/unique redirect → function
  package.json         # includes @netlify/blobs dependency
```

## Quick Deploy (Netlify CLI)
```
npm install -g netlify-cli
netlify login
netlify init                          # create or link a Netlify site
npm run deploy                        # deploy with functions + public/
```

## Git Deploy (optional)
```
git init
git add .
git commit -m "initial"
# create a repo online, then:
git remote add origin https://github.com/<you>/uap-orb-site.git
git push -u origin main
```
Netlify → **New site from Git** → select your repo
- Build command: (leave empty)
- Publish directory: `public`
- Functions directory: `netlify/functions`

When live, the badge calls `/api/unique` which redirects to the function.
