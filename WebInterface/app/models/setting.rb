class Setting < ApplicationRecord
    before_save :format_data

    def format_data
        if params[:setting][:models].present?
            self.models = params[:setting][:models].join(",")
        end
    end
end
