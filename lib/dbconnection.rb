require 'pg'

class DBConnection
  attr_accessor :db_name

  def initialize(db_name)
    @db_name = db_name
  end

  def connect
    connection = PG.connect(dbname: db_name)
    yield connection
    connection.finish
  end

  def calculate_probability_data
    sql = <<~SQL
    SELECT
    (SELECT value FROM token_stats WHERE name = 'category_count') AS category_count,
    (SELECT value FROM token_stats WHERE name = 'distinct_token_count') AS token_count,
    (SELECT value FROM token_stats WHERE name = 'total_token_count') AS total_example_count
    FROM token_stats LIMIT 1;
    SQL
    result = ''
    connect do |connection|
      result = connection.exec(sql).values[0].map { |num| num.to_f }
    end
    result
  end

  def upsert(category, token)
    sql = <<~SQL
    WITH upsert AS
    (UPDATE tokens SET frequency = frequency + 1 WHERE phrase = $2 AND
    category_id = (SELECT id FROM categories WHERE name = $1)
    RETURNING *)
    INSERT INTO tokens (phrase, frequency, category_id)
    SELECT $2, 1, (SELECT id FROM categories WHERE name = $1)
    WHERE NOT EXISTS (SELECT * FROM upsert);
    SQL
    connect { |connection| connection.exec_params(sql, [category, token]) }
  end

  def remove_from_category(category, token)
    sql = <<~SQL
    UPDATE tokens SET frequency = frequency - 1
    WHERE phrase = $2
    AND category_id = (SELECT id FROM categories WHERE name = $1);
    SQL
    connect { |connection| connection.exec_params(sql, [category, token]) }
  end

  def delete_from_category(category, token)
    sql = <<~SQL
    DELETE FROM tokens WHERE phrase = $2
    AND category_id = (SELECT id FROM categories WHERE name = $1);
    SQL
    connect { |connection| connection.exec_params(sql, [category, token]) }
  end
end

class DBToken < DBConnection
  def keys
    sql = "SELECT phrase FROM tokens;"
    result = ''
    connect { |connection| result = connection.exec(sql) }
    result.field_values('phrase')
  end

  def count
    sql = "SELECT value FROM token_stats WHERE name = 'distinct_token_count';"
    result = ''
    connect { |connection| result = connection.exec(sql) }
    result.values.first.first.to_i
  end

# possibly unused
  def update_frequency(phrase, frequency, category)
    sql = <<~SQL
    UPDATE tokens SET frequency = $2 WHERE phrase = $1
    AND category_id = (SELECT id FROM categories WHERE name = $3);
    SQL
    connect do |connection|
      connection.exec_params(sql, [phrase, frequency, category])
    end
  end

  def delete(token)
    sql = "DELETE FROM tokens WHERE phrase = $1;"
    connect { |connection| connection.exec_params(sql, [token]) }
  end
end

class DBData < DBConnection
  def [](category_name)
    sql = <<~SQL
    SELECT
    (SELECT value FROM token_stats WHERE name = 'token_frequency_count' AND category_id = (SELECT id FROM categories WHERE name = $1)) AS total_tokens,
    (SELECT value FROM token_stats WHERE name = 'token_count' AND category_id = (SELECT id FROM categories WHERE name = $1)) AS examples
    FROM token_stats LIMIT 1;
    SQL
    result = ''
    connect do |connection|
      result = connection.exec_params(sql, [category_name])
    end
    total_tokens = result.values[0][0].to_i
    examples = result.values[0][1].to_i
    { tokens: create_token_hash(category_name),
      total_tokens: total_tokens,
      examples: examples }
  end

  def create_token_hash(category)
    sql = <<~SQL
    SELECT phrase, frequency
    FROM tokens
    WHERE category_id = (SELECT id FROM categories WHERE name = $1);
    SQL
    token_hash = Hash.new
    result = ''
    connect { |connection| result = connection.exec_params(sql, [category]) }
    result.map do |tuple|
      token_hash[tuple['phrase']] = tuple['frequency'].to_i
    end
    token_hash
  end

  def keys
    sql = "SELECT name FROM categories;"
    result = ''
    connect { |connection| result = connection.exec(sql) }
    result.field_values('name')
  end

  def has_key?(value)
    sql = "SELECT name FROM categories WHERE name = $1;"
    result = ''
    connect { |connection| result = connection.exec_params(sql, [value]) }
    !result.values.empty?
  end

  def delete(category)
    tokens_sql = <<~SQL
    DELETE FROM tokens
    WHERE category_id = (SELECT id FROM categories WHERE name = $1);
    SQL
    connect { |connection| connection.exec_params(tokens_sql, [category]) }

    category_sql = "DELETE FROM categories WHERE name = $1;"
    connect { |connection| connection.exec_params(category_sql, [category]) }
  end
end
