
import csv

points = 0

num_channels = 7942
num_videos = 0
requests = 0

print "day 0: ", num_channels*12

with open('estimate.csv', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=',')

	# Estimates the quota costs for 200 days
	for i in range(1, 200):
		new_videos = num_channels * 0.27 # estimate videos per day and channel
		requests = num_channels * 2 # channel and playlist request
		requests += num_videos # known video request
		requests += new_videos # new videos request
		points = num_channels * 6 # fixed statistics costs

		points = points + num_videos * 3 + new_videos * 7 # costs

		num_videos = num_videos + new_videos

		print "day {}: {} - {} # {}".format(i, points, num_videos, requests)
		csvwriter.writerow([i, points, num_videos, requests])
		points = 0
		points_2 = 0

	print num_videos