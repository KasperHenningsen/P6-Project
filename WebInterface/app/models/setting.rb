class Setting < ApplicationRecord
  before_save :format_data

  validate :validate_datetime_fields

  def format_data
    if self.models.present?
      self.models = JSON.parse(self.models).reject(&:empty?).join(",")
    end
  end

  private

  def validate_datetime_fields
    unless DateTime.parse(self.start_date.to_s)
      errors.add(:start_date, "is not a valid datetime")
    end

    unless DateTime.parse(self.end_date.to_s)
      errors.add(:end_date, "is not a valid datetime")
    end

    if self.start_date.present? && self.end_date.present? && DateTime.parse(self.start_date.to_s) >= DateTime.parse(self.end_date.to_s)
      errors.add(:start_date, "should be before end date")
    end
  end
end
