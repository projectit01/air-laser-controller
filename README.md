# air-laser-controller

## Cấu hình

Chương trình đọc thiết lập mặc định từ `config.json`. Hai trường
`screen_width` và `screen_height` là độ rộng và chiều cao màn hình đích. Kết
quả JSON sẽ có thêm `screen_x` và `screen_y` sau khi quy đổi từ tọa độ camera.

`top_left` và `bottom_right` xác định khung TV trong ảnh camera:

```json
"top_left": [100, 50],
"bottom_right": [1180, 670]
```

Preview sẽ vẽ điểm trên-trái màu xanh, điểm dưới-phải màu đỏ và khung màu
vàng. Chỉ khi điểm laser nằm trong khung, chương trình mới trả về
`screen_x/screen_y`.

Có thể ghi đè cấu hình bằng tham số dòng lệnh, ví dụ:

```powershell
python laser_tracker.py --camera 0 --color red --screen-width 1920 --screen-height 1080
```

## Sử dụng Intel RealSense và lớp IR

Cài SDK Python:

```powershell
python -m pip install pyrealsense2
```

Trong `config.json`, đổi backend sang RealSense và chọn stream IR:

```json
{
  "source": "realsense",
  "color": "ir",
  "camera_width": 640,
  "camera_height": 480,
  "realsense_stream": "infrared",
  "realsense_index": 1,
  "realsense_fps": 30,
  "disable_emitter": true,
  "ir_min_value": 180
}
```

Sau đó chạy như bình thường:

```powershell
python laser_tracker.py
```

`realsense_index: 1` là camera IR trái; nếu cần camera IR phải, dùng `2`.
Ảnh IR là dữ liệu grayscale 8-bit, nên chương trình dùng `ir_min_value` thay
cho bộ lọc màu HSV. Có thể thử ghi đè nhanh:

```powershell
python laser_tracker.py --source realsense --color ir --no-preview
```

## Điều khiển mouse trực tiếp

Bật trong `config.json`:

```json
"mouse_control": true
```

Hoặc bật tạm thời bằng lệnh:

```powershell
python laser_tracker.py --mouse
```

Mouse chỉ di chuyển khi laser được phát hiện và nằm trong khung
`top_left/bottom_right`. Tắt bằng:

```powershell
python laser_tracker.py --no-mouse
```

Chức năng dùng Windows API nên không cần Unity hoặc thư viện mouse bổ sung.

Chương trình đọc camera, phát hiện điểm laser và trả về tọa độ pixel của điểm
laser trong khung hình.

## Cài đặt

```powershell
python -m pip install -r requirements.txt
```

## Chạy

```powershell
python laser_tracker.py
```

Mặc định chương trình dùng camera `0`, laser đỏ và mở hai cửa sổ preview.
Tọa độ được in ra dưới dạng JSON trên mỗi frame:

```json
{"detected": true, "x": 640, "y": 360, "normalized_x": 0.5, "normalized_y": 0.5}
```

Ví dụ:

```powershell
python laser_tracker.py --camera 1 --color green
python laser_tracker.py --no-preview
```

Nhấn `Q` hoặc `Esc` để dừng. `x`, `y` là tọa độ pixel trong ảnh camera,
không phải tọa độ màn hình. Bước tiếp theo là thêm hiệu chuẩn 4 góc màn hình
và phép biến đổi phối cảnh để đổi sang tọa độ màn hình.
