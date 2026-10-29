# SepFormer + JEPA

## Nội dung đã copy

- Các source/config SepFormer + JEPA đã xác định thuộc những nhánh `code`, `sepformer_jepa` và source-prediction được liệt kê bên dưới.
- Cây code thử nghiệm `research/work/test_DPRNN/Mamba-TasNet` đã được loại khỏi repo này vì đó là trainer/architecture Mamba-TasNet, không phải SepFormer. Các JEPA modules cần cho các source SepFormer + JEPA còn lại đã có bản trùng khớp trong `source/code/` và `source/sepformer_jepa/`. Các YAML còn trong `configs/test_DPRNN/` chỉ được lưu làm artifact tham khảo; chưa xác minh chúng với trainer tương ứng và không coi đó là pipeline được hỗ trợ.
- Source-prediction trainers/configs từ `source_predictive_sepformer` và `jepaformer`.
- Một LRS2 SepFormer YAML từ `research/Mamba-TasNet`; YAML này khai báo `jepa_weight: 0.1`, nhưng trainer tại source tree đó không có JEPA hooks. Nó được giữ trong `configs/unmatched_Mamba-TasNet/` để bảo toàn artifact và không được xem là baseline config hay một pipeline +JEPA đã xác minh.
- Các result folders riêng trong `artifacts/`, gồm checkpoint components, manifests, logs và JSON metrics có sẵn.

## Dataset

Không có audio/dataset nào được copy. Repo này chứa **cả Libri2Mix lẫn LRS2**, trong nhiều nhánh/config riêng:

| Variant/config hoặc run | Dataset | Path dataset ghi trong config/effective hyperparams | Ghi chú |
|---|---|---|---|
| Masked JEPA, `configs/sepformer_jepa/LibriMix/sepformer.yaml`; run `artifacts/sepformer_jepa-LibriMix-sepformer-1234/` | Libri2Mix `train-100/min` | Config source có `data_folder: !PLACEHOLDER`; effective run path: `/home/lab/letuan/code/speechseparation/data/LibriMix/storage/Libri2Mix` | Cần path này (hoặc chỉnh config) cho bản run tương ứng |
| Transformer JEPA, run `artifacts/*-sepformer_transformer_jepa-1234/` | Libri2Mix `train-100/min` | `/home/lab/letuan/code/speechseparation/data/LibriMix/storage/Libri2Mix` | Có hai result trees (`code` và `sepformer_jepa`); không gộp provenance |
| Masked JEPA LRS2, `configs/sepformer_jepa/LRS2/sepformer.yaml` | LRS2-2Mix | `/home/lab/letuan/code/speechseparation/data/LRS2-2Mix/lrs2_rebuild/audio` | Path của config branch này |
| `test_DPRNN` JEPA YAML trong thư mục LRS2 | YAML ghi `dataset: wsj0mix` nhưng đặt trong LRS2 | `data_folder: /project/anhlt/lab-ss/LRS2-2mix/lrs2_rebuild/audio` | Mâu thuẫn dataset label/path trong config; cần xác nhận/chỉnh trước khi dùng |
| Source-prediction, LRS2 config/run | Mixture LRS2-2Mix | Active static-mixture path: `data_folder: /project/anhlt/lab-ss/LRS2-2mix/lrs2_rebuild/audio`; YAML cũng khai báo `base_folder_dm: /project/anhlt/LibriMix/LibriSpeech/train-clean-100/` | `dynamic_mixing: false`; source code chỉ dùng `base_folder_dm` cho LibriMix dynamic-mixing branch, nên path LibriSpeech này không được dùng bởi config LRS2 đã copy |
| Source-prediction, Libri2Mix config/run | Mixture Libri2Mix | Active static-mixture path: `data_folder: /project/anhlt/LibriMix/Libri2Mix`; YAML cũng khai báo `base_folder_dm: /project/anhlt/LibriMix/LibriSpeech/train-clean-100/` | `dynamic_mixing: false`; `base_folder_dm` không được dùng cho dynamic mixing trong config/run đã copy |
| `jepaformer` source-prediction configs | Libri2Mix hoặc LRS2 static mixture | Libri2Mix `data_folder: /project/anhlt/LibriMix/Libri2Mix`; LRS2 `data_folder: /project/anhlt/lab-ss/LRS2-2mix/lrs2_rebuild/audio`; cả hai cũng khai báo `base_folder_dm: /project/anhlt/LibriMix/LibriSpeech/train-clean-100/` | YAML đặt `dynamic_mixing: false`; base path là khai báo dư/không active theo nhánh code hiện có. Source/config branch riêng, không có provenance nối chắc với run artifacts khác |
| LRS2 YAML lấy từ `research/Mamba-TasNet` | LRS2-2Mix | `/home/lab/letuan/code/speechseparation/data/LRS2-2Mix/lrs2_rebuild/audio` | YAML đặt `jepa_weight: 0.1`, nhưng trainer tree đó không có JEPA hooks; được giữ trong `configs/unmatched_Mamba-TasNet/`, không xác nhận pipeline |

Các `data_folder: !PLACEHOLDER` trong config là **chưa điền path**, không phải dataset path thực. Các path nêu trên chỉ là khai báo artifact; chưa xác nhận thư mục tồn tại ở máy đích. CSV trong run folders là manifests, không phải dữ liệu audio.

## Lưu ý

Đây là nhiều implementation/run khác nhau, không phải một model duy nhất. Một số log masked-JEPA kết thúc bằng `KeyboardInterrupt`. Không khẳng định run hoàn tất. Một số run folder có checkpoint optimizer lớn.
