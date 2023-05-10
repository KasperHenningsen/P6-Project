class Setting < ApplicationRecord
  before_save :format_data
  has_many :dataset, dependent: :destroy

  validate :validate_datetime_fields

  def format_data
    words = self.models.split(',').map { |word| word.gsub(/[^a-zA-Z ]/, '') }
    self.models = words.join(' ')
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

    if (self.end_date - self.start_date) > 10.years
      errors.add(:start_date, " and end date should not span more than 10 years")
    end
  end
end
