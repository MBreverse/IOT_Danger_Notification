# 展示物危險示警系統

通過IOT平台將終端裝置(樹梅派)偵測到的人體與展示物的距離回傳，並以明顯的圖示通知使用者。

執行方式:
1. 執行"camera_for_background.py"，令樹梅派的相機模組擷取展示物的背景
2. 執行"reference_object_location.py"，並手動框選畫面中的展示物
3. 執行"main_warning_detect.py", 開始向IOT平台傳輸資料並持續偵測環境中的人體與展示物的距離
