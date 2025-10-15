# 🎫 Xếp chỗ vòng tròn – Round-robin (Hamilton cycles)

Có một nhóm n đại biểu đến dự hội nghị cần “làm quen” nhau trong vài ngày bằng cách ngồi quanh một bàn tròn. Mỗi ngày, mọi người ngồi thành một vòng tròn; ai ngồi kề (trái/phải) nhau thì xem như đã làm quen. Mục tiêu là sắp xếp chỗ ngồi qua một số ngày sao cho:
1.	Trong từng ngày: mỗi người đều có hai bạn kề là người mới (không lặp lại người đã từng ngồi cạnh trước đó).
2.	Tổng thể sau D ngày: mọi cặp người trong nhóm đều đã từng ngồi kề ít nhất một lần (tức là ai cũng đã làm quen với tất cả những người còn lại).
Nhiệm vụ của bài toán là tạo lịch ngồi thỏa hai ý trên, càng ít ngày càng tốt, và có cách kiểm tra để chắc chắn lịch đó thật sự bao phủ đủ.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://support-tickets-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
