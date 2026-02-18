def trim_image(image, source_data_set, dest_data_set):
    my_vpos = int(dest_data_set.header1["HIERARCH SUBARRAY VPOS"]) if dest_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    my_hpos = int(dest_data_set.header1["HIERARCH SUBARRAY HPOS"]) if dest_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    bias_vpos = int(source_data_set.header1["HIERARCH SUBARRAY VPOS"]) if source_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    bias_hpos = int(source_data_set.header1["HIERARCH SUBARRAY HPOS"]) if source_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    start_x = my_vpos - bias_vpos
    start_y = my_hpos - bias_hpos
    return image[start_x:start_x + dest_data_set.image_shape[0], start_y:start_y + dest_data_set.image_shape[1]]