#!/bin/bash

read -p "Instance Number: " INST

if [ $INST -eq 1 ]; then
	gcloud compute ssh --ssh-flag="-L 8896:127.0.0.1:8896" --ssh-flag="-L 6006:127.0.0.1:6006" eric_pnr@instance-1 --zone=us-west1-a 
elif [ $INST -eq 2 ]; then
	gcloud compute ssh --ssh-flag="-L 8897:127.0.0.1:8896" --ssh-flag="-L 6007:127.0.0.1:6006" eric_pnr@instance-2 --zone=us-west2-a 
elif [ $INST -eq 3 ]; then
	gcloud compute ssh --ssh-flag="-L 8898:127.0.0.1:8896" --ssh-flag="-L 6008:127.0.0.1:6006" eric_pnr@instance-3 --zone=us-west2-a 
fi


