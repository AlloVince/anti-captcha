<?php
require_once('vendor/autoload.php');

use Gregwar\Captcha\CaptchaBuilder;
use Predis\Client;

function microtime_float()
{
    list($usec, $sec) = explode(" ", microtime());
    return ((float)$usec + (float)$sec);
}

$redis = new \Predis\Client([
    'scheme' => 'unix',
    'path' => '/tmp/redis.sock'
]);

$i = 1;
$MAX_KEYS = 10000;
$CHECK_INTERVAL = 10000;

while(true) {
    $start = microtime_float();
    $builder = new CaptchaBuilder;
    $builder->build(200, 80);
    $text = $builder->getPhrase();

    if ($i % $CHECK_INTERVAL === 0) {
        while(count($redis->keys('captcha:*')) > $MAX_KEYS) {
            sleep(5);
            echo "Redis keys more than $MAX_KEYS, sleeping\n";
        }
    }

    ob_start();
    $builder->output();
    $image = ob_get_contents();
    ob_end_clean();
    $redis->set("captcha:$text", $image);

    //$builder->save("samples/$text.jpg");
    $end = microtime_float();
    $diff = ($end - $start) * 1000;
    echo "[$i] Generated captcha:$text in $diff ms\n";
    //echo "[$i] Generated samples/$text.jpg in $diff ms\n";
    $i++;
}

