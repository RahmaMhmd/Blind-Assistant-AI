$cid = "1018753194837-rs6snne8l96hu8kj2j1sunjv0pkgdml1.apps.googleusercontent.com"
$sec = "GOCSPX-" + "ZYl65MCfftafIKQcKxcrZPhOlAyP"

python -m dvc remote modify --local myremote gdrive_client_id $cid
python -m dvc remote modify --local myremote gdrive_client_secret $sec
python -m dvc pull
