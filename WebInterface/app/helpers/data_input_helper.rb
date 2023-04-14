module DataInputHelper
  def convert_date(date)
    return Time.parse(date.to_s).strftime('%d-%m-%Y %H:%M')
  end

  def increment_date(date)
    return (Time.parse(date.to_s) + 1.hour)
  end

  def compare_dates(date, other_date)
    return Time.parse(date.to_s) <= Time.parse(other_date.to_s)
  end
end
