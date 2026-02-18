# Adding this module as a subtree

Go to the root of your repo you want to add this moduel to. Then,

```sh
mkdir lib
# make sure working tree is clean
git fetch && git pull
# merge latest files from main branch
git subtree add --prefix=lib/alc git@github.gatech.edu:SSOG/alc.git main
# push
git commit -am "Added ALC module as a subtree"
git push origin Branch_Name
```

Add this repo as a remote

```sh
git remote add -f alc_subtree git@github.gatech.edu:SSOG/alc.git
```

See all remotes to check if it is actually added

```sh
git remote -v
```

# Applying updates from this repo to your module

If you have not added this repo as remote on your machine, run (change the url as needed)

```sh
git remote add -f alc_subtree git@github.gatech.edu:SSOG/alc.git  # Or, use HTML url
git remote -v
```

Note `git clone`-ing your original/parent repo does not add the remote for this subtree.
The above operation is needed for every machine/environment for which you want this repo to be subtree.

Check for updates

```sh
git fetch alc_subtree main
```

Pull the changes if any

```sh
git fetch && git pull # as always
git subtree pull --prefix=lib/alc alc_subtree main --squash
```

`--squash` will pack all changes and commits from the subtree into one commit; remove it if you wish.

No need to `git add`; the changes are already commited and you just need to push the commit

```sh
git push
```
