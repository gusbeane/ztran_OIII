#!/bin/bash -x

# Force push the paper to GitHub
cd $TRAVIS_BUILD_DIR
git checkout --orphan $TRAVIS_BRANCH-pdf
git rm -rf .
git add -f paper/ms.pdf
git -c user.name='travis' -c user.email='travis' commit -m "building the paper"
git push -q -f https://$GITHUB_USER:$GITHUB_API_KEY@github.com/$TRAVIS_REPO_SLUG $TRAVIS_BRANCH-pdf

