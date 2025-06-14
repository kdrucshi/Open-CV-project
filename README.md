# Open-CV-project
This is a project based on Open CV library, it displays basic features and power of Open CV 
AIM - creating a program that can detect a hand, segment the hand, and count the number of fingers being held up!
Strategy -
- Our strategy is to grab a region of intrest and calculate running average background value for about 60 frames of video
- Once the value is calculated hand can enter region of intrest. 
- We apply thresholding after hand enters thus detecting change.
Counting fingers
- We create a convex hull around the hand once it enters region of intrest.

  ### ![hand_convex](https://github.com/user-attachments/assets/3da12dc2-61ea-42e3-b35f-fdbd4e70f525)
- After that we will calculate centre of the hand, and will you it to calculate distance to outer points at the edge of the hand.
- Thus, applying conditions we can calculate number of fingers that are up hence increasing counter.
![hand_convex](https://github.com/user-attachments/assets/3da12dc2-61ea-42e3-b35f-fdbd4e70f525)
