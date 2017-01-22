function [data_ret] = temp_fun(data,level)
data_ret = zeros(size(data));
if(mean(data)>level)
    data_ret = data;
end

end